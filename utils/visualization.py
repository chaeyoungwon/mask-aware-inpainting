import torchvision.utils as vutils
import torch


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def save_grid(masked, output, gt, path, nrow=4):
    imgs = torch.cat([denorm(masked), denorm(output), denorm(gt)], dim=0)
    vutils.save_image(imgs, path, nrow=nrow)
