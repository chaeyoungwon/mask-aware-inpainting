import torch
from torchvision.transforms.functional import rgb_to_grayscale


def psnr(output, gt, max_val=1.0):
    mse = ((output - gt) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse).item()


def ssim(output, gt, window_size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = torch.nn.functional.avg_pool2d(output, window_size, 1, window_size // 2)
    mu_y = torch.nn.functional.avg_pool2d(gt,     window_size, 1, window_size // 2)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sx  = torch.nn.functional.avg_pool2d(output ** 2, window_size, 1, window_size // 2) - mu_x2
    sy  = torch.nn.functional.avg_pool2d(gt     ** 2, window_size, 1, window_size // 2) - mu_y2
    sxy = torch.nn.functional.avg_pool2d(output * gt, window_size, 1, window_size // 2) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sxy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sx + sy + C2)
    return num.div(den).mean().item()
