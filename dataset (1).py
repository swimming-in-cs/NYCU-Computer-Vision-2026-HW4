"""
Dataset for HW4 Image Restoration (Rain / Snow).

Directory structure expected:
    data/
        train/
            degraded/
                rain-1.png  ... rain-1600.png
                snow-1.png  ... snow-1600.png
            clean/
                rain_clean-1.png  ... rain_clean-1600.png
                snow_clean-1.png  ... snow_clean-1600.png
        test/
            degraded/
                0.png ... 99.png
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class RestoreTrainDataset(Dataset):
    """
    Loads paired (degraded, clean) images from both rain and snow categories.
    Applies random crop + augmentations for training.
    """

    def __init__(self, data_root, patch_size=128):
        self.patch_size = patch_size
        self.pairs = []

        degraded_dir = Path(data_root) / 'train' / 'degraded'
        clean_dir    = Path(data_root) / 'train' / 'clean'

        # Rain
        for i in range(1, 1601):
            deg  = degraded_dir / f'rain-{i}.png'
            cln  = clean_dir    / f'rain_clean-{i}.png'
            if deg.exists() and cln.exists():
                self.pairs.append((str(deg), str(cln)))

        # Snow
        for i in range(1, 1601):
            deg  = degraded_dir / f'snow-{i}.png'
            cln  = clean_dir    / f'snow_clean-{i}.png'
            if deg.exists() and cln.exists():
                self.pairs.append((str(deg), str(cln)))

        assert len(self.pairs) > 0, \
            f"No training pairs found under {data_root}/train/. Check folder structure."
        print(f"[Dataset] {len(self.pairs)} training pairs loaded.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_path, cln_path = self.pairs[idx]
        deg = Image.open(deg_path).convert('RGB')
        cln = Image.open(cln_path).convert('RGB')

        # --- Random crop ---
        i, j, h, w = self._get_random_crop_params(deg, self.patch_size)
        deg = TF.crop(deg, i, j, h, w)
        cln = TF.crop(cln, i, j, h, w)

        # --- Augmentations ---
        if random.random() > 0.5:
            deg = TF.hflip(deg)
            cln = TF.hflip(cln)
        if random.random() > 0.5:
            deg = TF.vflip(deg)
            cln = TF.vflip(cln)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            deg = TF.rotate(deg, angle)
            cln = TF.rotate(cln, angle)

        deg = TF.to_tensor(deg)  # [0, 1]
        cln = TF.to_tensor(cln)
        return deg, cln

    @staticmethod
    def _get_random_crop_params(img, output_size):
        w, h = img.size
        th = tw = output_size
        if w < tw or h < th:
            # Upscale if needed (shouldn't happen with this dataset)
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


# ---------------------------------------------------------------------------
# Validation Dataset
# ---------------------------------------------------------------------------

class RestoreValDataset(Dataset):
    """
    Same as train but without augmentations; uses center crop for consistency.
    Re-uses a fixed subset of training pairs as validation.
    """

    def __init__(self, data_root, patch_size=128, val_ratio=0.05):
        train_ds = RestoreTrainDataset(data_root, patch_size)
        n_val = max(1, int(len(train_ds.pairs) * val_ratio))
        # Take the last n_val pairs as validation
        self.pairs = train_ds.pairs[-n_val:]
        self.patch_size = patch_size
        print(f"[Dataset] {len(self.pairs)} validation pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_path, cln_path = self.pairs[idx]
        deg = Image.open(deg_path).convert('RGB')
        cln = Image.open(cln_path).convert('RGB')

        deg = TF.center_crop(deg, self.patch_size)
        cln = TF.center_crop(cln, self.patch_size)

        return TF.to_tensor(deg), TF.to_tensor(cln)


# ---------------------------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------------------------

class RestoreTestDataset(Dataset):
    """Loads test images (0.png … 99.png) without ground truth."""

    def __init__(self, data_root):
        test_dir = Path(data_root) / 'test' / 'degraded'
        self.image_paths = sorted(
            test_dir.glob('*.png'),
            key=lambda p: int(p.stem)
        )
        assert len(self.image_paths) > 0, \
            f"No test images found in {test_dir}. Check folder structure."
        print(f"[Dataset] {len(self.image_paths)} test images loaded.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        name = path.name  # e.g. '0.png'
        return TF.to_tensor(img), name
