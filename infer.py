import os, sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RestoreTestDataset
from net.model import PromptIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',    type=str, required=True)
    parser.add_argument('--checkpoint',   type=str, required=True)
    parser.add_argument('--output',       type=str, default='pred.npz')
    parser.add_argument('--tile_size',    type=int, default=256)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--no_tta',       action='store_true', help='Disable 8-fold TTA')
    return parser.parse_args()


def tile_infer(model, img, tile, overlap, device):
    _, C, H, W = img.shape
    if tile <= 0 or (H <= tile and W <= tile):
        with torch.no_grad():
            return model(img.to(device)).clamp(0, 1).cpu()
    stride = tile - overlap
    out   = torch.zeros_like(img)
    count = torch.zeros(1, 1, H, W)
    ys = sorted(set(list(range(0, H-tile+1, stride)) + ([H-tile] if H > tile else [])))
    xs = sorted(set(list(range(0, W-tile+1, stride)) + ([W-tile] if W > tile else [])))
    for y in ys:
        for x in xs:
            t = img[:, :, y:y+tile, x:x+tile].to(device)
            with torch.no_grad():
                o = model(t).clamp(0, 1).cpu()
            out[:, :, y:y+tile, x:x+tile]   += o
            count[:, :, y:y+tile, x:x+tile] += 1
    return out / count.clamp(min=1)


def tta_forward(model, img, tile, overlap, device):
    """8-fold D4 TTA: 4 rotations × (original + hflip)"""
    preds = []
    for flip in [False, True]:
        for rot in [0, 1, 2, 3]:
            x = img.clone()
            if flip:
                x = torch.flip(x, dims=[-1])
            if rot > 0:
                x = torch.rot90(x, k=rot, dims=[-2, -1])
            out = tile_infer(model, x, tile, overlap, device)
            if rot > 0:
                out = torch.rot90(out, k=-rot, dims=[-2, -1])
            if flip:
                out = torch.flip(out, dims=[-1])
            preds.append(out)
    return torch.stack(preds).mean(0)


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = PromptIR(prompt=True, prompt_len=10).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()
    print(f'Loaded checkpoint (epoch {ckpt.get("epoch","?")})')

    test_dataset = RestoreTestDataset(args.data_root)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    results = {}
    for i, (img_tensor, (name,)) in enumerate(test_loader):
        if args.no_tta:
            out = tile_infer(model, img_tensor, args.tile_size, args.tile_overlap, device)
        else:
            out = tta_forward(model, img_tensor, args.tile_size, args.tile_overlap, device)
        # np.rint: 四捨五入，避免 floor 的系統性 -0.5 誤差
        arr = np.rint(out.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
        results[name] = arr
        print(f'[{i+1}/{len(test_loader)}] {name}: {arr.shape}')

    np.savez(args.output, **results)
    print(f'\nSaved {len(results)} images to {args.output}')


if __name__ == '__main__':
    main()
