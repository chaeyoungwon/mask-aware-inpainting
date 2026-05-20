import os
import torch
from config import Config
from datasets.celeba_dataset import get_celeba_loader
from models.cae import CAE
from utils.losses import InpaintingLoss
from utils.visualization import save_grid


def train():
    cfg = Config()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"device: {device}")
    os.makedirs(cfg.save_dir, exist_ok=True)

    loader = get_celeba_loader(cfg.data_root, 'train', cfg.batch_size,
                               cfg.image_size, cfg.num_workers,
                               max_samples=cfg.max_train_samples)

    model = CAE(base_ch=cfg.base_ch).to(device)
    criterion = InpaintingLoss(perceptual_weight=cfg.w_perceptual).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.num_epochs):
        model.train()
        for step, (masked, mask, gt) in enumerate(loader):
            masked, mask, gt = masked.to(device), mask.to(device), gt.to(device)

            output = model(masked, mask)
            loss, logs = criterion(output, gt, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % cfg.log_every == 0:
                print(f"[{epoch+1}/{cfg.num_epochs}] step {step} | "
                      f"loss {loss.item():.4f} | "
                      f"valid {logs['l_valid'].item():.4f} | "
                      f"hole {logs['l_hole'].item():.4f} | "
                      f"perc {logs['l_perc']:.4f}")

        save_grid(masked[:4], output[:4], gt[:4],
                  f"{cfg.save_dir}/epoch{epoch+1:03d}.png")
        torch.save(model.state_dict(), f"{cfg.save_dir}/epoch{epoch+1:03d}.pth")


if __name__ == '__main__':
    train()
