"""
prepare_fixed_testset.py
────────────────────────
Pre-generate a fixed test set (gt / mask / masked) from the CelebA test split
and save each sample as individual .pt files.  Run this ONCE before evaluating
any model so all three models (Vanilla / PConv / Gated) see identical images
and masks.

Usage
─────
    python prepare_fixed_testset.py
    python prepare_fixed_testset.py --n 1000 --out ./fixed_testset --seed 42

Output
──────
    fixed_testset/
    ├── 0000_gt.pt       — ground-truth tensor (3, H, W), range [0, 1]
    ├── 0000_mask.pt     — binary mask tensor  (1, H, W), hole=0 valid=1
    ├── 0000_masked.pt   — gt * mask           (3, H, W)
    ├── 0001_gt.pt
    ...
"""

import argparse
import os
import random

import numpy as np
import torch

from config import Config
from datasets.celeba_dataset import CelebAInpaintingDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',    type=int, default=500,
                        help='Number of samples to save (default: 500)')
    parser.add_argument('--out',  type=str, default='./fixed_testset',
                        help='Output directory (default: ./fixed_testset)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for reproducibility (default: 42)')
    args = parser.parse_args()

    cfg = Config()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    dataset = CelebAInpaintingDataset(
        root=cfg.data_root,
        split='test',
        image_size=cfg.image_size,
        max_samples=args.n,
    )

    n = len(dataset)
    print(f"Saving {n} samples to {args.out}/  (seed={args.seed}) ...")

    hole_ratios = []
    for i in range(n):
        masked, mask, gt = dataset[i]
        idx = f"{i:04d}"
        torch.save(gt,     os.path.join(args.out, f"{idx}_gt.pt"))
        torch.save(mask,   os.path.join(args.out, f"{idx}_mask.pt"))
        torch.save(masked, os.path.join(args.out, f"{idx}_masked.pt"))
        hole_ratios.append((1.0 - mask.mean()).item())

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}")

    arr = np.array(hole_ratios)
    print(f"\nSaved {n} samples → {args.out}/")
    print(f"Hole ratio — mean: {arr.mean():.3f} | min: {arr.min():.3f} | max: {arr.max():.3f}")
    print(f"  small  (< 0.3)      : {(arr < 0.3).sum()} samples")
    print(f"  medium (0.3 – 0.5)  : {((arr >= 0.3) & (arr < 0.5)).sum()} samples")
    print(f"  large  (0.5 – 0.7)  : {((arr >= 0.5) & (arr < 0.7)).sum()} samples")


if __name__ == '__main__':
    main()
