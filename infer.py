"""
infer.py — Run inference and generate pred.npz for submission (HW4)

Usage:
    python infer.py \
        --data_root ./data \
        --checkpoint ./checkpoints/best_model.pth \
        --output pred.npz

The output pred.npz contains:
    keys  : filename (e.g. '0.png', '1.png', ...)
    values: uint8 numpy array of shape (3, H, W) — RGB
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from net.model import PromptIR
from dataset import RestoreTestDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for HW4 Image Restoration')
    parser.add_argument('--data_root',   type=str, required=True)
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--output',      type=str, default='pred.npz')
    parser.add_argument('--tile_size',   type=int, default=256,
                        help='Tile size for large image inference (0 = no tiling)')
    parser.add_argument('--tile_overlap',type=int, default=32,
                        help='Overlap between tiles in pixels')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tiled Inference
# ---------------------------------------------------------------------------

def tile_inference(model, img_tensor, tile_size, overlap, device):
    """
    Run model on overlapping tiles and stitch results.
    Useful for large images to avoid OOM.
    """
    _, C, H, W = img_tensor.shape
    if tile_size <= 0 or (H <= tile_size and W <= tile_size):
        return model(img_tensor.to(device)).clamp(0, 1).cpu()

    stride = tile_size - overlap
    output = torch.zeros_like(img_tensor)
    count  = torch.zeros(1, 1, H, W)

    ys = list(range(0, H - tile_size + 1, stride)) + ([H - tile_size] if H > tile_size else [])
    xs = list(range(0, W - tile_size + 1, stride)) + ([W - tile_size] if W > tile_size else [])
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    for y in ys:
        for x in xs:
            tile = img_tensor[:, :, y:y+tile_size, x:x+tile_size].to(device)
            with torch.no_grad():
                out_tile = model(tile).clamp(0, 1).cpu()
            output[:, :, y:y+tile_size, x:x+tile_size] += out_tile
            count[:, :, y:y+tile_size, x:x+tile_size]  += 1

    output = output / count.clamp(min=1)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Infer] Using device: {device}")

    # Load model
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[Infer] Checkpoint loaded from {args.checkpoint}")

    # Test dataset
    test_dataset = RestoreTestDataset(args.data_root)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    results = {}

    with torch.no_grad():
        for img_tensor, (name,) in tqdm(test_loader, desc='Inference'):
            out = tile_inference(model, img_tensor, args.tile_size, args.tile_overlap, device)
            # Convert to uint8 numpy (3, H, W)
            out_np = (out.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
            results[name] = out_np
            print(f"  {name}: shape={out_np.shape}")

    # Save
    np.savez(args.output, **results)
    print(f"\n[Done] Saved {len(results)} images to {args.output}")
    print("  Keys:", list(results.keys())[:5], "...")
    print("  Sample shape:", next(iter(results.values())).shape)


if __name__ == '__main__':
    main()
