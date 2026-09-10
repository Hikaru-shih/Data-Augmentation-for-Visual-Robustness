import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torchvision import transforms

from src.augmentations.cutmix import cutmix_data
from src.augmentations.mixup import mixup_data, mixup_loss
from src.datasets.cifar10 import get_train_transform
from src.datasets.cifar10c import CIFAR10C
from src.datasets.splits import split_indices
from src.utils.runs import reserve_run, file_hash, verify_evaluation, validate_summary
from scripts.evaluate_corruptions import CORRUPTIONS
from scripts import train as training


class PipelineTests(unittest.TestCase):
    def test_training_selects_validation_then_tests_once(self):
        config = dict(experiment={"name": "baseline"}, seed=42,
                      dataset={"name": "cifar10", "num_classes": 10}, model={"name": "resnet18"},
                      optimizer={"name": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 0.0005},
                      scheduler={"name": "cosine"}, training={"epochs": 3}, augmentation={"name": "baseline"})
        with tempfile.TemporaryDirectory() as root:
            result, checkpoints = Path(root) / "results", Path(root) / "checkpoints"
            result.mkdir()
            checkpoints.mkdir()
            model = torch.nn.Linear(1, 10)
            def fake_train(model, loader, criterion, optimizer, device):
                optimizer.zero_grad()
                model.weight.sum().backward()
                optimizer.step()
                return 1.0, 20.0
            with patch("sys.argv", ["train", "--config", "unused"]), patch.object(training, "load_config", return_value=config), patch.object(training, "reserve_run", return_value=("test", result, checkpoints)), patch.object(training, "training_loaders", return_value=("train", "validation", {})), patch.object(training, "test_loader", return_value="test") as test_factory, patch.object(training, "get_resnet18", return_value=model), patch.object(training, "train_one_epoch", side_effect=fake_train), patch.object(training, "evaluate", side_effect=[(1, 60), (1, 80), (1, 70), (1, 75)]) as evaluation:
                training.main()
            saved = torch.load(checkpoints / "best_model.pt", weights_only=True)
            self.assertEqual(saved["epoch"], 2)
            self.assertEqual(saved["test_accuracy"], 75)
            self.assertEqual([call.args[1] for call in evaluation.call_args_list], ["validation"] * 3 + ["test"])
            test_factory.assert_called_once()

    def test_split_is_fixed_disjoint_and_complete(self):
        train, val = split_indices(100, 10, 2026)
        self.assertEqual((train, val), split_indices(100, 10, 2026))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train + val), set(range(100)))
        self.assertEqual(len(val), 10)

    def test_augmentation_parameters_reach_transform(self):
        pipeline = get_train_transform("randaugment", num_ops=1, magnitude=4)
        aug = next(t for t in pipeline.transforms if isinstance(t, transforms.RandAugment))
        self.assertEqual((aug.num_ops, aug.magnitude), (1, 4))
        pipeline = get_train_transform("augmix", severity=2, mixture_width=2, chain_depth=1, alpha=0.5)
        aug = next(t for t in pipeline.transforms if isinstance(t, transforms.AugMix))
        self.assertEqual((aug.severity, aug.mixture_width, aug.chain_depth, aug.alpha), (2, 2, 1, 0.5))
        with self.assertRaises(ValueError):
            get_train_transform("randaugment", typo=3)

    def test_mixup_values_and_loss(self):
        images = torch.stack([torch.zeros(1, 4, 4), torch.ones(1, 4, 4)])
        labels = torch.tensor([0, 1])
        with patch("numpy.random.beta", return_value=0.25), patch("torch.randperm", return_value=torch.tensor([1, 0])):
            mixed, a, b, lam = mixup_data(images, labels)
        self.assertTrue(torch.allclose(mixed, 0.25 * images + 0.75 * images.flip(0)))
        logits = torch.tensor([[2., -1.], [-1., 2.]])
        ce = torch.nn.CrossEntropyLoss()
        self.assertTrue(torch.allclose(mixup_loss(ce, logits, a, b, lam), 0.25 * ce(logits, labels) + 0.75 * ce(logits, labels.flip(0))))

    def test_cutmix_actual_area_and_no_mutation(self):
        images = torch.stack([torch.zeros(1, 4, 6), torch.ones(1, 4, 6)])
        original = images.clone()
        with patch("src.augmentations.cutmix.rand_bbox", return_value=(0, 1, 3, 3)), patch("torch.randperm", return_value=torch.tensor([1, 0])):
            mixed, _, _, lam = cutmix_data(images, torch.tensor([0, 1]))
        self.assertEqual(lam, 0.75)
        self.assertEqual(mixed[0].sum().item(), 6)
        self.assertTrue(torch.equal(images, original))

    def test_cifar10c_severity_and_labels(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            pixels = np.lib.format.open_memmap(root / "fog.npy", mode="w+", dtype=np.uint8, shape=(50000, 32, 32, 3))
            pixels[:] = 0
            pixels[20000:30000] = 127
            del pixels
            np.save(root / "labels.npy", np.repeat(np.arange(5), 10000))
            dataset = CIFAR10C(root, "fog", 3)
            self.assertEqual(len(dataset), 10000)
            self.assertEqual(int(dataset.images[0, 0, 0, 0]), 127)
            self.assertEqual(dataset[0][1], 2)
            np.save(root / "labels.npy", np.zeros(9999, dtype=int))
            with self.assertRaises(ValueError):
                CIFAR10C(root, "fog", 1)

    def test_run_collision_and_stale_evaluation(self):
        with tempfile.TemporaryDirectory() as root, patch("src.utils.runs.Path", side_effect=lambda p: Path(root) / p):
            key, result, checkpoints = reserve_run("baseline", 42, "test")
            with self.assertRaises(FileExistsError):
                reserve_run("baseline", 42, "test")
            checkpoint = checkpoints / "best_model.pt"
            checkpoint.write_bytes(b"first")
            directory = result / "robustness"
            directory.mkdir()
            rows = [dict(corruption=name, mean_accuracy=50) for name in CORRUPTIONS + ["overall_mean"]]
            summary = directory / "corruption_summary.csv"
            with summary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["corruption", "mean_accuracy"])
                writer.writeheader()
                writer.writerows(rows)
            detail = directory / "corruption_results.csv"
            detail.write_text("test")
            (directory / "metadata.json").write_text(json.dumps(dict(checkpoint_sha256=file_hash(checkpoint), files={p.name: file_hash(p) for p in (summary, detail)})))
            verify_evaluation(key)
            checkpoint.write_bytes(b"second")
            with self.assertRaises(ValueError):
                verify_evaluation(key)
            with self.assertRaises(ValueError):
                validate_summary(rows[:-1])


if __name__ == "__main__":
    unittest.main()
