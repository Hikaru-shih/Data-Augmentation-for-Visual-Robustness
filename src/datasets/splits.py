import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from src.datasets.cifar10 import get_train_transform, get_test_transform


def split_indices(size, validation_size, split_seed):
    if not 0 < validation_size < size:
        raise ValueError("validation_size must be between 1 and dataset size - 1")
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(split_seed)).tolist()
    return indices[validation_size:], indices[:validation_size]


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def training_loaders(config):
    aug = dict(config["augmentation"])
    name = aug.pop("name")
    if name in {"mixup", "cutmix"}:
        if set(aug) - ({"alpha", "probability"} if name == "cutmix" else {"alpha"}):
            raise ValueError("Unknown mixing parameter")
        name, aug = "baseline", {}
    root = config["dataset"]["data_dir"]
    split = config["validation"]
    if config["dataset"]["name"] == "cifar100":
        from src.datasets.cifar100 import prepare
        train, validation, train_ids, val_ids = prepare(config, name, aug)
    else:
        train = CIFAR10(root, train=True, download=True, transform=get_train_transform(name, **aug))
        validation = CIFAR10(root, train=True, download=False, transform=get_test_transform())
        train_ids, val_ids = split_indices(len(train), split["size"], split["seed"])
    kwargs = dict(batch_size=config["training"]["batch_size"],
                  num_workers=config["training"].get("num_workers", 4),
                  pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    train_loader = DataLoader(Subset(train, train_ids), shuffle=True,
                              generator=torch.Generator().manual_seed(config["seed"]), **kwargs)
    val_loader = DataLoader(Subset(validation, val_ids), shuffle=False,
                            generator=torch.Generator().manual_seed(split["seed"]), **kwargs)
    return train_loader, val_loader, {"train": train_ids, "validation": val_ids}


def test_loader(config):
    if config["dataset"]["name"] == "cifar100":
        from torchvision.datasets import CIFAR100
        from src.datasets.cifar100 import test_transform
        dataset = CIFAR100(config["dataset"]["data_dir"], train=False, download=True, transform=test_transform(config))
    else:
        dataset = CIFAR10(config["dataset"]["data_dir"], train=False, download=True,
                          transform=get_test_transform())
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                      num_workers=config["training"].get("num_workers", 4))
