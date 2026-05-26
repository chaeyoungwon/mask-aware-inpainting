"""
prepare_fixed_testset.py
────────────────────────
Pre-generate a fixed set of masks for the test split and save them as a
numpy array.  Run this ONCE before evaluating any model so that all three
models (Vanilla / PConv / Gated) are compared under identical mask conditions.

Usage
─────
    python prepare_fixed_testset.py                     # 500 masks, default path
    python prepare_fixed_testset.py --n 1000 --out ./data/fixed_test_masks.npy

Output
──────
    ./data/fixed_test_masks.npy   — float32 array, shape (N, 1, 128, 128)
                                    hole=0, valid=1  (same convention as mask_generator)
"""

import argparse
import os
import random

import numpy as np
import torch

from config import Config
from datasets.mask_generator import generate_stroke_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',   type=int, default=500,
                        help='Number of masks to generate (default: 500)')
    parser.add_argument('--out', type=str, default='',
                        help='Output .npy path (default: <data_root>/fixed_test_masks.npy)')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for reproducibility (default: 0)')
    args = parser.parse_args()

    cfg = Config()

    out_path = args.out or os.path.join(cfg.data_root, 'fixed_test_masks.npy')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Fix seed so the same masks are always generated
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Generating {args.n} fixed masks  (image_size={cfg.image_size}, seed={args.seed}) ...")
    masks = []
    for i in range(args.n):
        mask = generate_stroke_mask(cfg.image_size, cfg.image_size)  # (1, H, W) tensor
        masks.append(mask.numpy())
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.n}")

    arr = np.stack(masks, axis=0)   # (N, 1, H, W)
    np.save(out_path, arr)

    hole_ratios = 1.0 - arr.mean(axis=(1, 2, 3))
    print(f"\nSaved → {out_path}")
    print(f"Shape : {arr.shape}   dtype: {arr.dtype}")
    print(f"Hole ratio — mean: {hole_ratios.mean():.3f} | "
          f"min: {hole_ratios.min():.3f} | max: {hole_ratios.max():.3f}")
    print(f"  small  (<0.2)  : {(hole_ratios < 0.2).sum()} masks")
    print(f"  medium (0.2–0.4): {((hole_ratios >= 0.2) & (hole_ratios < 0.4)).sum()} masks")
    print(f"  large  (>=0.4) : {(hole_ratios >= 0.4).sum()} masks")


if __name__ == '__main__':
    main()
