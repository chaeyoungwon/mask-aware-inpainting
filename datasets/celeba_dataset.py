import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets.mask_generator import generate_stroke_mask


class CelebAInpaintingDataset(Dataset):
    def __init__(self, root, split="train", image_size=64, max_samples=None):
        # root가 ./data 또는 ./data/celeba 등일 때 이미지 폴더를 자동 탐색
        candidates = [
            root,
            os.path.join(root, 'celeba'),
            os.path.join(root, 'img_align_celeba'),
            os.path.join(root, 'celeba', 'img_align_celeba'),
        ]

        self.root = None
        self.image_paths = []
        for path in candidates:
            if not os.path.isdir(path):
                continue
            files = [
                fname for fname in os.listdir(path)
                if fname.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if files:
                self.root = path
                self.image_paths = sorted(os.path.join(path, fname) for fname in files)
                break

        if self.root is None:
            self.root = root
            self.image_paths = [
                os.path.join(root, fname)
                for fname in os.listdir(root)
                if fname.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in: {self.root}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        # mask: visible=1, missing=0
        mask = generate_stroke_mask(self.image_size, self.image_size)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask = mask.float()
        masked_image = image * mask

        return masked_image, mask, image


def get_celeba_loader(root, split, batch_size, image_size=64, num_workers=0, max_samples=None):
    dataset = CelebAInpaintingDataset(
        root=root,
        split=split,
        image_size=image_size,
        max_samples=max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader