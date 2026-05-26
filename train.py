import os
import csv
import random

import numpy as np
import torch
from config import Config
from datasets.celeba_dataset import get_celeba_loader
from utils.losses import InpaintingLoss
from utils.metrics import psnr, ssim
from utils.visualization import save_grid


# ──────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
#  Model factory (shared with evaluate.py)
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
#  Early Stopping
# ──────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training when val_psnr hasn't improved for `patience` epochs.
    Returns True from .step() when a new best is found.
    """
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        improved = (self.best is None) or (score > self.best + self.min_delta)
        if improved:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


# ──────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────
def train():
    cfg = Config()
    set_seed(cfg.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"device: {device}  |  conv_type: {cfg.conv_type}  |  seed: {cfg.seed}")

    save_dir = os.path.join(cfg.save_dir, cfg.conv_type)
    os.makedirs(save_dir, exist_ok=True)

    # ── Data ──
    train_loader = get_celeba_loader(
        cfg.data_root, 'train', cfg.batch_size,
        cfg.image_size, cfg.num_workers,
        max_samples=cfg.max_train_samples,
    )
    val_loader = get_celeba_loader(
        cfg.data_root, 'valid', cfg.batch_size,
        cfg.image_size, cfg.num_workers,
        max_samples=2000,   # keep validation fast
    )

    # ── Model ──
    model     = build_model(cfg).to(device)
    criterion = InpaintingLoss(perceptual_weight=cfg.w_perceptual).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Scheduler ──
    scheduler = None
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min
        )

    # ── Early stopping ──
    early_stop = EarlyStopping(patience=cfg.early_stopping_patience)
    best_psnr  = -float('inf')

    # ── CSV history log ──
    csv_path = os.path.join(save_dir, f'training_history_{cfg.conv_type}.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'val_loss', 'val_psnr', 'val_ssim', 'lr']
        )

    # ── Training loop ──
    for epoch in range(cfg.num_epochs):

        # ── Train ──
        model.train()
        train_loss_sum, train_steps = 0.0, 0

        for step, (masked, mask, gt) in enumerate(train_loader):
            masked, mask, gt = masked.to(device), mask.to(device), gt.to(device)

            output = model(masked, mask)
            loss, logs = criterion(output, gt, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps    += 1

            if step % cfg.log_every == 0:
                print(f"  [{epoch+1}/{cfg.num_epochs}] step {step:4d} | "
                      f"loss {loss.item():.4f} | "
                      f"valid {logs['l_valid'].item():.4f} | "
                      f"hole {logs['l_hole'].item():.4f} | "
                      f"perc {logs['l_perc']:.4f}")

        train_loss_avg = train_loss_sum / train_steps

        # ── Validate ──
        model.eval()
        val_loss_sum = val_psnr_sum = val_ssim_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for masked, mask, gt in val_loader:
                masked, mask, gt = masked.to(device), mask.to(device), gt.to(device)
                output = model(masked, mask)
                loss, _ = criterion(output, gt, mask)
                val_loss_sum += loss.item()
                val_psnr_sum += psnr(output, gt)
                val_ssim_sum += ssim(output, gt)
                val_steps    += 1

        val_loss_avg = val_loss_sum / val_steps
        val_psnr_avg = val_psnr_sum / val_steps
        val_ssim_avg = val_ssim_sum / val_steps
        current_lr   = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch+1:3d}/{cfg.num_epochs}] "
              f"train_loss: {train_loss_avg:.4f} | "
              f"val_loss: {val_loss_avg:.4f} | "
              f"val_PSNR: {val_psnr_avg:.2f} | "
              f"val_SSIM: {val_ssim_avg:.4f} | "
              f"lr: {current_lr:.2e}")

        # ── Scheduler step (after val so lr logged before update) ──
        if scheduler is not None:
            scheduler.step()

        # ── Save sample grid ──
        save_grid(masked[:4], output[:4], gt[:4],
                  f"{save_dir}/epoch{epoch+1:03d}.png")

        # ── Append to CSV ──
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"{train_loss_avg:.6f}",
                f"{val_loss_avg:.6f}",
                f"{val_psnr_avg:.4f}",
                f"{val_ssim_avg:.6f}",
                f"{current_lr:.2e}",
            ])

        # ── Best model save + early stopping ──
        improved = early_stop.step(val_psnr_avg)
        if improved:
            best_psnr = val_psnr_avg
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_model.pth'))
            print(f"  → Best model saved  (val_PSNR: {best_psnr:.2f})")

        if early_stop.should_stop:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {cfg.early_stopping_patience} epochs)")
            break

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best val_PSNR : {best_psnr:.2f} dB")
    print(f"  Best model    : {os.path.join(save_dir, 'best_model.pth')}")
    print(f"  History CSV   : {csv_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    train()
