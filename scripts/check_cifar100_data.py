"""Check archive structure and label identity; do not score or train a model."""
from pathlib import Path
import numpy as np
from torchvision.datasets import CIFAR100
from scripts.evaluate_corruptions import CORRUPTIONS


def main():
    root = Path("data/CIFAR-100-C")
    missing = [name for name in ["labels.npy"] + [c + ".npy" for c in CORRUPTIONS] if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Extract CIFAR-100-C into {root}; missing: {', '.join(missing)}")
    labels = np.load(root / "labels.npy")
    test = CIFAR100("data", train=False, download=True)
    expected = np.asarray(test.targets)
    if labels.shape == (50000,):
        expected = np.tile(expected, 5)
    if not np.array_equal(labels, expected):
        raise ValueError("Expected CIFAR-100 fine test labels in matching order")
    for name in CORRUPTIONS:
        images = np.load(root / f"{name}.npy", mmap_mode="r")
        if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError(f"Invalid corruption array: {name}")
    print("CIFAR-100-C structure and labels verified; no model evaluation performed.")


if __name__ == "__main__":
    main()
