import os, sys, time, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import RestoreTrainDataset, RestoreValDataset
from net.model import PromptIR
from utils.losses import CombinedLoss
from utils.metrics import compute_psnr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',    type=str,   default='./data')
    parser.add_argument('--save_dir',     type=str,   default='./checkpoints')
    parser.add_argument('--epochs',       type=int,   default=150)
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--patch_size',   type=int,   default=128)
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--lambda_fft',   type=float, default=0.05)
    parser.add_argument('--val_freq',     type=int,   default=5)
    parser.add_argument('--num_workers',  type=int,   default=2)
    parser.add_argument('--resume',       type=str,   default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_dataset = RestoreTrainDataset(args.data_root, args.patch_size)
    val_dataset   = RestoreValDataset(args.data_root,   args.patch_size)
    train_loader  = DataLoader(train_dataset, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader    = DataLoader(val_dataset, 1, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    model     = PromptIR(prompt=True, prompt_len=10).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    criterion = CombinedLoss(args.lambda_fft)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=1e-4)
    warmup    = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup, eta_min=1e-6)

    train_losses, val_psnrs, val_epochs = [], [], []
    start_epoch, best_psnr = 1, 0.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch  = ckpt['epoch'] + 1
        best_psnr    = ckpt.get('best_psnr', 0.0)
        train_losses = ckpt.get('train_losses', [])
        val_psnrs    = ckpt.get('val_psnrs', [])
        val_epochs   = ckpt.get('val_epochs', [])
        print(f'Resumed from epoch {ckpt["epoch"]}, PSNR={best_psnr:.2f}')

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss, t0 = 0.0, time.time()

        if epoch <= warmup:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * epoch / warmup

        for deg, cln in train_loader:
            deg, cln = deg.to(device), cln.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(deg)
                loss, _, _ = criterion(out, cln)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        if epoch > warmup:
            scheduler.step()

        avg = epoch_loss / len(train_loader)
        train_losses.append(avg)
        print(f'Epoch [{epoch}/{args.epochs}] Loss={avg:.4f} | '
              f'LR={optimizer.param_groups[0]["lr"]:.2e} | {time.time()-t0:.0f}s')

        if epoch % args.val_freq == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                psnr = sum(compute_psnr(model(d.to(device)).clamp(0, 1), c.to(device))
                           for d, c in val_loader) / len(val_loader)
            val_psnrs.append(psnr)
            val_epochs.append(epoch)
            print(f'  [Val] PSNR={psnr:.2f} dB')

            ckpt_data = {
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'train_losses': train_losses, 'val_psnrs': val_psnrs, 'val_epochs': val_epochs,
            }
            if psnr > best_psnr:
                best_psnr = psnr
                ckpt_data['best_psnr'] = best_psnr
                torch.save(ckpt_data, os.path.join(args.save_dir, 'best_model.pth'))
                print(f'  ✓ Best model saved! PSNR={best_psnr:.2f}')
            torch.save(ckpt_data, os.path.join(args.save_dir, 'latest_model.pth'))

            # Save learning curve
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(range(1, len(train_losses)+1), train_losses, color='steelblue')
            ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss'); ax1.grid(True)
            ax2.plot(val_epochs, val_psnrs, color='coral', marker='o')
            ax2.set_xlabel('Epoch'); ax2.set_ylabel('PSNR (dB)')
            ax2.set_title('Validation PSNR'); ax2.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'learning_curve.png'), dpi=150)
            plt.close()

    print(f'Done! Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
