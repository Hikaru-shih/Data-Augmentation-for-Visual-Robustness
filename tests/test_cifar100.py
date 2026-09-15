import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import yaml
from PIL import Image
from src.datasets.cifar100 import stratified_indices, training_statistics, prepare, test_transform
from src.datasets.cifar10c import CIFAR10C
from src.datasets.splits import training_loaders
from src.models.resnet import get_resnet18


class TinyCIFAR100:
    def __init__(self, root, train=True, download=False, transform=None):
        self.targets = np.repeat(np.arange(100), 3).tolist()
        self.data = np.random.default_rng(0).integers(0, 256, (300, 32, 32, 3), dtype=np.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        return self.transform(image), self.targets[index]


class CIFAR100Tests(unittest.TestCase):
    def test_experiment_names_match_batch_and_comparison(self):
        for method in ("baseline", "cutmix", "cutmix_p05"):
            config = yaml.safe_load(Path(f"configs/cifar100_{method}.yaml").read_text())
            self.assertEqual(config["experiment"]["name"], f"cifar100_{method}_resnet18")
            self.assertEqual(config["dataset"]["name"], "cifar100")
            self.assertEqual(config["dataset"]["num_classes"], 100)

    def test_stratified_split(self):
        labels = np.repeat(np.arange(100), 5)
        train, val = stratified_indices(labels, 2, 2026)
        self.assertEqual((train, val), stratified_indices(labels, 2, 2026))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train + val), set(range(500)))
        np.testing.assert_array_equal(np.bincount(labels[val]), np.full(100, 2))

    def test_statistics_exclude_holdout(self):
        images = np.array([np.zeros((2, 2, 3)), np.full((2, 2, 3), 255), np.full((2, 2, 3), 127)], dtype=np.uint8)
        stats = training_statistics(images, [0, 1])
        np.testing.assert_allclose(stats["mean"], [.5] * 3)
        np.testing.assert_allclose(stats["std"], [.5] * 3)
        images[2] = 0
        self.assertEqual(stats, training_statistics(images, [0, 1]))

    def test_loaders_normalization_and_forward_backward(self):
        config = yaml.safe_load(Path("configs/cifar100_cutmix_p05.yaml").read_text())
        config["validation"]["size"] = 100
        config["training"].update(num_workers=0, batch_size=2)
        with patch("src.datasets.cifar100.CIFAR100", TinyCIFAR100):
            train, val, ids = training_loaders(config)
        self.assertEqual(len(ids["train"]), 200)
        self.assertEqual(len(ids["validation"]), 100)
        self.assertEqual(config["dataset"]["normalization"]["source"], "training_split")
        image = Image.fromarray(train.dataset.dataset.data[0])
        self.assertTrue(torch.equal(val.dataset.dataset.transform(image), test_transform(config)(image)))
        model = get_resnet18(100)
        inputs, labels = next(iter(train))
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2, 100))
        torch.nn.CrossEntropyLoss()(logits, labels).backward()
        self.assertIsNotNone(model.fc.weight.grad)

    def test_corruption_accepts_99_and_rejects_100(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            images = np.lib.format.open_memmap(path / "fog.npy", mode="w+", dtype=np.uint8, shape=(50000, 32, 32, 3))
            images[20000] = 255
            del images
            np.save(path / "labels.npy", np.full(10000, 99, dtype=np.int64))
            dataset = CIFAR10C(path, "fog", 3, num_classes=100, mean=[.5]*3, std=[.5]*3)
            image, label = dataset[0]
            self.assertEqual(label, 99)
            self.assertTrue(torch.equal(image, torch.ones(3,32,32)))
            del dataset
            np.save(path / "labels.npy", np.full(10000, 100, dtype=np.int64))
            with self.assertRaises(ValueError):
                CIFAR10C(path, "fog", num_classes=100)
