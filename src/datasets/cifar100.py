"""CIFAR-100 fine labels, fixed stratified holdout and train-only statistics."""
import hashlib
import numpy as np
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from src.datasets.cifar10 import get_train_transform


def stratified_indices(targets, per_class, seed, classes=100):
    targets = np.asarray(targets)
    if set(targets.tolist()) != set(range(classes)) or per_class < 1:
        raise ValueError("Expected all fine classes and a positive holdout size")
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label in range(classes):
        indices = np.flatnonzero(targets == label)
        if len(indices) <= per_class:
            raise ValueError("Each class must retain training examples")
        indices = indices[torch.randperm(len(indices), generator=generator).numpy()].tolist()
        validation.extend(indices[:per_class])
        train.extend(indices[per_class:])
    return train, validation


def training_statistics(images, indices):
    total = np.zeros(3, dtype=np.float64)
    squares = total.copy()
    pixels = 0
    if not indices:
        raise ValueError("Empty training split")
    for start in range(0, len(indices), 256):
        batch = images[indices[start:start + 256]].astype(np.float64) / 255.0
        total += batch.sum(axis=(0, 1, 2))
        squares += (batch * batch).sum(axis=(0, 1, 2))
        pixels += np.prod(batch.shape[:3])
    mean = total / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean * mean, 0))
    if np.any(std <= 0):
        raise ValueError("Degenerate normalization")
    return dict(mean=mean.tolist(), std=std.tolist(), source="training_split", ddof=0)


def normalization(config):
    norm = config["dataset"]["normalization"]
    if norm.get("source") != "training_split" or norm.get("ddof") != 0:
        raise ValueError("CIFAR-100 requires frozen train-only population statistics")
    if len(norm["mean"]) != 3 or len(norm["std"]) != 3 or not np.isfinite(norm["mean"] + norm["std"]).all() or min(norm["std"]) <= 0:
        raise ValueError("Invalid normalization")
    return transforms.Normalize(norm["mean"], norm["std"])


def test_transform(config):
    return transforms.Compose([transforms.ToTensor(), normalization(config)])


def prepare(config, name, parameters):
    root = config["dataset"]["data_dir"]
    train = CIFAR100(root, train=True, download=True)
    split = config["validation"]
    if split.get("strategy") != "stratified" or split["size"] % 100:
        raise ValueError("CIFAR-100 needs stratified holdout divisible by 100")
    train_ids, val_ids = stratified_indices(train.targets, split["size"] // 100, split["seed"])
    config["dataset"]["normalization"] = training_statistics(train.data, train_ids)
    digest = hashlib.sha256(train.data.tobytes())
    digest.update(np.asarray(train.targets, dtype="<i8").tobytes())
    config["dataset"]["training_data_sha256"] = digest.hexdigest()
    pipeline = get_train_transform(name, **parameters)
    pipeline.transforms[-1] = normalization(config)
    train.transform = pipeline
    validation = CIFAR100(root, train=True, download=False, transform=test_transform(config))
    return train, validation, train_ids, val_ids
