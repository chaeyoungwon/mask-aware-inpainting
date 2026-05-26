"""
evaluate.py
───────────
Evaluate a trained inpainting model on the CelebA test split.

Features
────────
• Fixed-mask support  — load pre-saved masks so all models see identical damage
• Full-image metrics  — PSNR / SSIM over the whole reconstructed image
• Hole-only metrics   — PSNR and L1 computed only inside the masked region
• Hole-ratio buckets  — small (<0.2) / medium (0.2–0.4) / large (≥0.4)
• CSV summary         — results saved to <checkpoint_dir>/eval_results.csv
• Visual grid         — first batch saved as eval_<conv_type>_sample.png

Usage
─────
    python evaluate.py checkpoints/gated/best_model.pth
    python evaluate.py checkpoints/vanilla/best_model.pth --fixed_masks ./data/fixed_test_masks.npy
"""

import os
import sys
import csv
import argparse
from collections import defaultdict

import torch
import numpy as np

from config import Config
from datasets.celeba_dataset import get_celeba_loader
from utils.metrics import psnr, ssim
from utils.visualization import save_grid


# ──────────────────────────────────────────────
#  Model factory (mirrors train.py)
# ──────────────────────────────────────────────
def build_model(cfg):
    if cfg.conv_type == 'vanilla':
        from models.cae import CAE
        return CAE(base_ch=cfg.base_ch)
    elif cfg.conv_type == 'pconv':
        from models.pconv_cae import PConvCAE
        return PConvCAE(base_ch=cfg.base_ch)
    elif cfg.conv_type == 'gated':
        from models.gated_cae import GatedCAE
        return GatedCAE(base_ch=cfg.base_ch)
    else:
        raise ValueError(
            f"Unknown conv_type: {cfg.conv_type!r}. "
            "Choose 'vanilla', 'pconv', or 'gated'."
        )


# ──────────────────────────────────────────────
#  Hole-only metrics
# ──────────────────────────────────────────────
def hole_psnr(output: torch.Tensor, gt: torch.Tensor,
              mask: torch.Tensor, max_val: float = 1.0) -> float:
    """
    PSNR computed only inside the hole region (mask == 0).
    output, gt : (B, C, H, W), values in [-1, 1]  (normalized)
    mask       : (B, 1, H, W), hole=0 valid=1
    """
    hole = (mask == 0).expand_as(output)
    if hole.sum() == 0:
        return float('nan')
    diff   = (output[hole] - gt[hole]) ** 2
    mse    = diff.mean().item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def hole_l1(output: torch.Tensor, gt: torch.Tensor,
            mask: torch.Tensor) -> float:
    """Mean absolute error inside the hole region."""
    hole = (mask == 0).expand_as(output)
    if hole.sum() == 0:
        return float('nan')
    return (output[hole] - gt[hole]).abs().mean().item()


def hole_ratio(mask: torch.Tensor) -> float:
    """Fraction of pixels that are holes (mask == 0), averaged over batch."""
    return (1.0 - mask.float()).mean().item()


def ratio_bucket(ratio: float) -> str:
    if ratio < 0.2:
        return 'small'
    elif ratio < 0.4:
        return 'medium'
    else:
        return 'large'


# ──────────────────────────────────────────────
#  Main evaluation
# ──────────────────────────────────────────────
def evaluate(checkpoint_path: str, fixed_masks_path: str = None):
    cfg = Config()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"device: {device}  |  conv_type: {cfg.conv_type}")
    print(f"checkpoint: {checkpoint_path}")

    # ── Fixed masks ──
    fixed_masks = None
    if fixed_masks_path:
        fixed_masks = np.load(fixed_masks_path)
        print(f"Fixed masks loaded: {fixed_masks.shape}  from {fixed_masks_path}")
    else:
        # Fallback: look for default path
        default_path = os.path.join(cfg.data_root, 'fixed_test_masks.npy')
        if os.path.exists(default_path):
            fixed_masks = np.load(default_path)
            print(f"Fixed masks auto-loaded: {fixed_masks.shape}  from {default_path}")
        else:
            print("WARNING: No fixed masks found — using random masks (not fair for comparison!)")
            print("         Run prepare_fixed_testset.py first, or pass --fixed_masks <path>")

    loader = get_celeba_loader(
        cfg.data_root, 'test', cfg.batch_size,
        cfg.image_size, cfg.num_workers,
        fixed_masks=fixed_masks,
    )

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Accumulators ──
    total = defaultdict(float)
    counts = defaultdict(int)          # per bucket
    bucket_psnr = defaultdict(float)
    bucket_hole_psnr = defaultdict(float)
    bucket_ssim = defaultdict(float)

    with torch.no_grad():
        for i, (masked, mask, gt) in enumerate(loader):
            masked, mask, gt = masked.to(device), mask.to(device), gt.to(device)
            output = model(masked, mask)

            # ── Full-image metrics ──
            b_psnr = psnr(output, gt)
            b_ssim = ssim(output, gt)

            # ── Hole-only metrics ──
            b_hole_psnr = hole_psnr(output, gt, mask)
            b_hole_l1   = hole_l1(output, gt, mask)
            b_ratio     = hole_ratio(mask)
            bucket       = ratio_bucket(b_ratio)

            total['psnr']      += b_psnr
            total['ssim']      += b_ssim
            total['hole_psnr'] += b_hole_psnr
            total['hole_l1']   += b_hole_l1
            total['count']     += 1

            bucket_psnr[bucket]      += b_psnr
            bucket_hole_psnr[bucket] += b_hole_psnr
            bucket_ssim[bucket]      += b_ssim
            counts[bucket]           += 1

            # ── Visual grid for first batch ──
            if i == 0:
                out_img = os.path.join(
                    os.path.dirname(checkpoint_path),
                    f"eval_{cfg.conv_type}_sample.png"
                )
                save_grid(masked[:4], output[:4], gt[:4], out_img)
                print(f"Sample grid saved → {out_img}")

    n = total['count']
    avg_psnr      = total['psnr']      / n
    avg_ssim      = total['ssim']      / n
    avg_hole_psnr = total['hole_psnr'] / n
    avg_hole_l1   = total['hole_l1']   / n

    # ── Print summary ──
    sep = '─' * 56
    print(f"\n{sep}")
    print(f"  Model      : {cfg.conv_type}   ({n_params:,} params)")
    print(f"  Batches    : {n}")
    print(sep)
    print(f"  Full-image PSNR  : {avg_psnr:.2f} dB")
    print(f"  Full-image SSIM  : {avg_ssim:.4f}")
    print(f"  Hole-only  PSNR  : {avg_hole_psnr:.2f} dB")
    print(f"  Hole-only  L1    : {avg_hole_l1:.4f}")
    print(sep)
    print(f"  Per-bucket (hole ratio):")
    for b in ['small', 'medium', 'large']:
        if counts[b] > 0:
            print(f"    {b:7s} (<{'0.2' if b=='small' else '0.4' if b=='medium' else '1.0'}) "
                  f"| n={counts[b]:4d} "
                  f"| PSNR={bucket_psnr[b]/counts[b]:.2f} "
                  f"| hole_PSNR={bucket_hole_psnr[b]/counts[b]:.2f} "
                  f"| SSIM={bucket_ssim[b]/counts[b]:.4f}")
    print(sep)

    # ── Save CSV ──
    csv_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f"eval_results_{cfg.conv_type}.csv"
    )
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['model',      cfg.conv_type])
        w.writerow(['n_params',   n_params])
        w.writerow(['n_batches',  n])
        w.writerow(['psnr',       f"{avg_psnr:.4f}"])
        w.writerow(['ssim',       f"{avg_ssim:.6f}"])
        w.writerow(['hole_psnr',  f"{avg_hole_psnr:.4f}"])
        w.writerow(['hole_l1',    f"{avg_hole_l1:.6f}"])
        for b in ['small', 'medium', 'large']:
            if counts[b] > 0:
                w.writerow([f'psnr_{b}',      f"{bucket_psnr[b]/counts[b]:.4f}"])
                w.writerow([f'hole_psnr_{b}', f"{bucket_hole_psnr[b]/counts[b]:.4f}"])
                w.writerow([f'ssim_{b}',      f"{bucket_ssim[b]/counts[b]:.6f}"])
                w.writerow([f'n_{b}',         counts[b]])

    print(f"  CSV saved  → {csv_path}")
    print(sep + "\n")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a trained inpainting model.'
    )
    parser.add_argument('checkpoint', help='Path to best_model.pth')
    parser.add_argument('--fixed_masks', default=None,
                        help='Path to fixed_test_masks.npy (optional; '
                             'auto-detected from data_root if omitted)')
    args = parser.parse_args()
    evaluate(args.checkpoint, fixed_masks_path=args.fixed_masks)
