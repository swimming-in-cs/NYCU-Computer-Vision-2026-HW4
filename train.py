"""
train.py — Train PromptIR for Rain/Snow Image Restoration (HW4)

Usage:
    python train.py --data_root ./data --epochs 150 --batch_size 4

Google Colab tips:
    - Use --batch_size 2 if OOM on T4
    - Enable GPU: Runtime → Change runtime type → GPU
"""

import os
import argparse
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import RestoreTrainDataset, RestoreValDataset
from net.model import PromptIR
from utils.losses import CombinedLoss
from utils.metrics import compute_psnr


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train PromptIR for Image Restoration')
    parser.add_argument('--data_root',   type=str,   default='./data')
    parser.add_argument('--save_dir',    type=str,   default='./checkpoints')
    parser.add_argument('--epochs',      type=int,   default=150)
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--patch_size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--lambda_fft',  type=float, default=0.05,
                        help='Weight for FFT frequency loss')
    parser.add_argument('--resume',      type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--val_freq',    type=int,   default=5,
                        help='Validate every N epochs')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Using device: {device}")

    # ---- Datasets & DataLoaders ----
    train_dataset = RestoreTrainDataset(args.data_root, patch_size=args.patch_size)
    val_dataset   = RestoreValDataset(args.data_root,   patch_size=args.patch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ---- Model ----
    model = PromptIR(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        prompt=True,
        prompt_len=10,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {total_params / 1e6:.2f}M")

    # ---- Loss & Optimizer ----
    criterion = CombinedLoss(lambda_fft=args.lambda_fft)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=1e-4)

    # Cosine Annealing with warm-up (10 epochs)
    warmup_epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs,
        eta_min=1e-6,
    )

    start_epoch = 1
    best_psnr   = 0.0

    # ---- Resume ----
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', 0.0)
        print(f"[Resume] Epoch {ckpt['epoch']}, best PSNR: {best_psnr:.2f}")

    # ---- Training ----
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        # Warm-up: linearly increase LR
        if epoch <= warmup_epochs:
            warmup_lr = args.lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        for step, (degraded, clean) in enumerate(train_loader, 1):
            degraded = degraded.to(device, non_blocking=True)
            clean    = clean.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                restored = model(degraded)
                loss, l1_loss, fft_loss = criterion(restored, clean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % 100 == 0:
                print(f"  Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (L1: {l1_loss.item():.4f}, "
                      f"FFT: {fft_loss.item():.4f})")

        if epoch > warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        print(f"[Epoch {epoch}/{args.epochs}] Avg Loss: {avg_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s")

        # ---- Validation ----
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            psnr = validate(model, val_loader, device)
            print(f"[Val] Epoch {epoch} | PSNR: {psnr:.2f} dB")

            # Save best
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_psnr': best_psnr,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"  ✓ Best model saved! PSNR: {best_psnr:.2f} dB")

            # Save latest
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }, os.path.join(args.save_dir, 'latest_model.pth'))

    print(f"\n[Done] Best PSNR: {best_psnr:.2f} dB")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    psnr_total = 0.0
    for degraded, clean in val_loader:
        degraded = degraded.to(device)
        clean    = clean.to(device)
        restored = model(degraded).clamp(0, 1)
        psnr_total += compute_psnr(restored, clean)
    return psnr_total / len(val_loader)


if __name__ == '__main__':
    main()
