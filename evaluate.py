"""
evaluate.py
───────────
Evaluate a trained inpainting model on the CelebA test split.

Features
────────
• Fixed-testset support — load pre-saved gt/mask/masked .pt files
• Full-image metrics    — PSNR / SSIM over the whole reconstructed image
• Hole-only metrics     — PSNR and L1 computed only inside the masked region
• Hole-ratio buckets    — small (0.1–0.3) / medium (0.3–0.5) / large (0.5–0.7)
• CSV summary           — results saved to <checkpoint_dir>/eval_results.csv
• Visual grid           — first batch saved as eval_<conv_type>_sample.png

Usage
─────
    python evaluate.py checkpoints/vanilla/best_model.pth --conv_type vanilla --fixed_testset ./fixed_testset
    python evaluate.py checkpoints/pconv/best_model.pth   --conv_type pconv   --fixed_testset ./fixed_testset
    python evaluate.py checkpoints/gated/best_model.pth   --conv_type gated   --fixed_testset ./fixed_testset
"""

import os
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
    """PSNR inside the hole region (mask == 0)."""
    hole = (mask == 0).expand_as(output)
    if hole.sum() == 0:
        return float('nan')
    mse = ((output[hole] - gt[hole]) ** 2).mean().item()
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
    if ratio < 0.3:
        return 'small'
    elif ratio < 0.5:
        return 'medium'
    else:
        return 'large'


# ──────────────────────────────────────────────
#  Fixed testset loader
# ──────────────────────────────────────────────
def load_fixed_testset(fixed_testset_dir: str, batch_size: int):
    """
    Yield (masked, mask, gt) CPU-tensor batches from pre-saved .pt files.
    Expects files named: 0000_gt.pt / 0000_mask.pt / 0000_masked.pt
    """
    files = sorted(f for f in os.listdir(fixed_testset_dir) if f.endswith('_gt.pt'))
    if not files:
        raise FileNotFoundError(
            f"No *_gt.pt files found in {fixed_testset_dir}. "
            "Run prepare_fixed_testset.py first."
        )
    for start in range(0, len(files), batch_size):
        batch = files[start:start + batch_size]
        gts, masks, maskeds = [], [], []
        for fname in batch:
            idx = fname.split('_')[0]
            gts.append(torch.load(os.path.join(fixed_testset_dir, f"{idx}_gt.pt")))
            masks.append(torch.load(os.path.join(fixed_testset_dir, f"{idx}_mask.pt")))
            maskeds.append(torch.load(os.path.join(fixed_testset_dir, f"{idx}_masked.pt")))
        yield torch.stack(maskeds), torch.stack(masks), torch.stack(gts)


# ──────────────────────────────────────────────
#  Main evaluation
# ──────────────────────────────────────────────
def evaluate(checkpoint_path: str, fixed_testset_dir: str = None, conv_type: str = None):
    cfg = Config()
    if conv_type is not None:
        cfg.conv_type = conv_type

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"device: {device}  |  conv_type: {cfg.conv_type}")
    print(f"checkpoint: {checkpoint_path}")

    # ── Data source ──
    if fixed_testset_dir:
        print(f"Fixed testset: {fixed_testset_dir}")
        data_iter = load_fixed_testset(fixed_testset_dir, cfg.batch_size)
    else:
        print("WARNING: No fixed testset — using random masks (not fair for comparison!)")
        print("         Run prepare_fixed_testset.py first, or pass --fixed_testset <dir>")
        data_iter = get_celeba_loader(
            cfg.data_root, 'test', cfg.batch_size,
            cfg.image_size, cfg.num_workers,
        )

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Accumulators ──
    total = defaultdict(float)
    counts = defaultdict(int)
    bucket_psnr      = defaultdict(float)
    bucket_hole_psnr = defaultdict(float)
    bucket_ssim      = defaultdict(float)

    with torch.no_grad():
        for i, (masked, mask, gt) in enumerate(data_iter):
            masked, mask, gt = masked.to(device), mask.to(device), gt.to(device)
            output = model(masked, mask)

            b_psnr      = psnr(output, gt)
            b_ssim      = ssim(output, gt)
            b_hole_psnr = hole_psnr(output, gt, mask)
            b_hole_l1   = hole_l1(output, gt, mask)

            total['psnr']      += b_psnr
            total['ssim']      += b_ssim
            total['hole_psnr'] += b_hole_psnr
            total['hole_l1']   += b_hole_l1
            total['count']     += 1

            for j in range(output.size(0)):
                out_j  = output[j:j+1]
                gt_j   = gt[j:j+1]
                mask_j = mask[j:j+1]
                r = 1.0 - mask_j.float().mean().item()
                bucket = ratio_bucket(r)
                bucket_psnr[bucket]      += psnr(out_j, gt_j)
                bucket_hole_psnr[bucket] += hole_psnr(out_j, gt_j, mask_j)
                bucket_ssim[bucket]      += ssim(out_j, gt_j)
                counts[bucket]           += 1

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
    bucket_ranges = {'small': '0.1–0.3', 'medium': '0.3–0.5', 'large': '0.5–0.7'}
    for b in ['small', 'medium', 'large']:
        if counts[b] > 0:
            print(f"    {b:7s} ({bucket_ranges[b]}) "
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
        w.writerow(['model',     cfg.conv_type])
        w.writerow(['n_params',  n_params])
        w.writerow(['n_batches', n])
        w.writerow(['psnr',      f"{avg_psnr:.4f}"])
        w.writerow(['ssim',      f"{avg_ssim:.6f}"])
        w.writerow(['hole_psnr', f"{avg_hole_psnr:.4f}"])
        w.writerow(['hole_l1',   f"{avg_hole_l1:.6f}"])
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
    parser.add_argument('--conv_type', choices=['vanilla', 'pconv', 'gated'], required=True,
                        help='Model type to evaluate')
    parser.add_argument('--fixed_testset', default=None,
                        help='Directory containing pre-saved *_gt.pt / *_mask.pt / *_masked.pt '
                             'files (created by prepare_fixed_testset.py)')
    args = parser.parse_args()
    evaluate(args.checkpoint, fixed_testset_dir=args.fixed_testset, conv_type=args.conv_type)
