import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import Dataset, DataLoader, Subset
from datasets.mask_generator import generate_stroke_mask


class CelebAInpaintingDataset(Dataset):
    """
    CelebA dataset wrapped for inpainting.

    Parameters
    ----------
    root        : dataset root (CelebA will be downloaded here if needed)
    split       : 'train' | 'valid' | 'test'
    image_size  : resize target (square)
    max_samples : optionally cap the number of samples
    fixed_masks : numpy array of shape (N, 1, H, W) with pre-saved masks.
                  If provided, mask for sample i is fixed_masks[i % N].
                  Use this for the test split to ensure fair model comparison.
    """

    def __init__(self, root='./data', split='train', image_size=256,
                 max_samples=None, fixed_masks=None):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        celeba = CelebA(root=root, split=split, transform=transform, download=False)
        if max_samples is not None:
            celeba = Subset(celeba, range(min(max_samples, len(celeba))))
        self.celeba      = celeba
        self.image_size  = image_size
        self.fixed_masks = fixed_masks  # (N, 1, H, W) numpy array or None

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        gt, _ = self.celeba[idx]   # ground truth (C, H, W), values in [0, 1]

        if self.fixed_masks is not None:
            # Use pre-saved mask for this index (wrap-around if needed)
            mask = torch.from_numpy(
                self.fixed_masks[idx % len(self.fixed_masks)]
            ).float()              # (1, H, W), hole=0 valid=1
        else:
            mask = generate_stroke_mask(self.image_size, self.image_size)

        masked_image = gt * mask   # zero-fill the hole
        return masked_image, mask, gt


def get_celeba_loader(root='./data', split='train', batch_size=16,
                      image_size=256, num_workers=4, max_samples=None,
                      fixed_masks=None):
    """
    Parameters
    ----------
    fixed_masks : path to .npy file (str) or numpy array.
                  If a string is passed, the array is loaded from that path.
                  Passed through to CelebAInpaintingDataset.
    """
    if isinstance(fixed_masks, str):
        fixed_masks = np.load(fixed_masks)

    dataset = CelebAInpaintingDataset(
        root=root, split=split, image_size=image_size,
        max_samples=max_samples, fixed_masks=fixed_masks,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
    )
    return loader
