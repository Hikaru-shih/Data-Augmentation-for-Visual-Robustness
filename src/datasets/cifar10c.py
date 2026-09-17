from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.cifar10 import CIFAR10_MEAN, CIFAR10_STD


class CIFAR10C(Dataset):

    def __init__(
        self,
        root,
        corruption,
        severity=1,
        num_classes=10,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    ):
        if severity < 1 or severity > 5:
            raise ValueError(
                "Severity must be between 1 and 5."
            )

        root = Path(root)

        corruption_path = root / f"{corruption}.npy"
        labels_path = root / "labels.npy"

        if not corruption_path.exists():
            raise FileNotFoundError(
                f"Corruption file not found: {corruption_path}"
            )

        images = np.load(corruption_path, mmap_mode="r")
        labels = np.load(labels_path)
        if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError("Expected uint8 CIFAR-10-C images of shape (50000, 32, 32, 3)")
        if labels.shape not in {(10000,), (50000,)} or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Expected 10000 or 50000 integer labels")
        if np.any((labels < 0) | (labels >= num_classes)):
            raise ValueError(f"Labels must be in [0, {num_classes - 1}]")

        start = (severity - 1) * 10000
        end = severity * 10000

        self.images = images[start:end]

        # CIFAR-10-C labels may contain labels for all corruption
        # severity blocks. Select the matching range when necessary.
        if len(labels) == 50000:
            self.labels = labels[start:end]
        else:
            self.labels = labels[:10000]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean,
                std,
            ),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])

        image = Image.fromarray(image)
        image = self.transform(image)

        return image, label
