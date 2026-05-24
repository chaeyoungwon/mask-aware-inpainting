from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import Dataset, DataLoader, Subset
from datasets.mask_generator import generate_stroke_mask


class CelebAInpaintingDataset(Dataset):
    def __init__(self, root='./data', split='train', image_size=256, max_samples=None):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        celeba = CelebA(root=root, split=split, transform=transform, download=True)
        if max_samples is not None:
            celeba = Subset(celeba, range(min(max_samples, len(celeba))))
        self.celeba = celeba
        self.image_size = image_size

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        gt, _ = self.celeba[idx]                         # ground truth (C, H, W)
        mask = generate_stroke_mask(self.image_size, self.image_size)  # (1, H, W), hole=0
        masked_image = gt * mask                         # hole 부분을 0으로
        return masked_image, mask, gt


def get_celeba_loader(root='./data', split='train', batch_size=16, image_size=256, num_workers=4, max_samples=None):
    dataset = CelebAInpaintingDataset(root=root, split=split, image_size=image_size, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
    )
    return loader
