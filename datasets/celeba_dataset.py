import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets.mask_generator import create_center_mask, create_random_block_mask


class CelebAInpaintingDataset(Dataset):
    def __init__(self, root, split="train", image_size=64, max_samples=None):
        self.root = root

        self.image_paths = [
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_paths = sorted(self.image_paths)

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in: {root}")

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
        mask = create_random_block_mask(self.image_size, self.image_size)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask = mask.float()

        masked_image = image * mask
        hole_mask = 1 - mask

        return {
            "image": image,
            "masked_image": masked_image,
            "mask": mask,
            "hole_mask": hole_mask,
        }


def get_celeba_loader(root, split, batch_size, image_size=64, max_samples=None):
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
        num_workers=2,
        pin_memory=True,
    )

    return loader