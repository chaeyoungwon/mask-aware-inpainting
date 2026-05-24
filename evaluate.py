import torch
from config import Config
from datasets.celeba_dataset import get_celeba_loader
from utils.metrics import psnr, ssim
from utils.visualization import save_grid


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
        raise ValueError(f"Unknown conv_type: {cfg.conv_type!r}. Choose 'vanilla', 'pconv', or 'gated'.")


def evaluate(checkpoint_path):
    cfg = Config()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"device: {device}  |  conv_type: {cfg.conv_type}")

    loader = get_celeba_loader(cfg.data_root, 'test', cfg.batch_size,
                               cfg.image_size, cfg.num_workers)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    total_psnr, total_ssim, count = 0.0, 0.0, 0

    with torch.no_grad():
        for i, (masked, mask, gt) in enumerate(loader):
            masked, mask, gt = masked.to(device), mask.to(device), gt.to(device)
            output = model(masked, mask)

            total_psnr += psnr(output, gt)
            total_ssim += ssim(output, gt)
            count += 1

            if i == 0:
                save_grid(masked[:4], output[:4], gt[:4],
                          f"eval_{cfg.conv_type}_sample.png")

    print(f"[{cfg.conv_type}] PSNR: {total_psnr/count:.2f} | SSIM: {total_ssim/count:.4f}")


if __name__ == '__main__':
    import sys
    evaluate(sys.argv[1])
