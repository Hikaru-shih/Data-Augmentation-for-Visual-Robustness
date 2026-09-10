import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)


def get_train_transform(augmentation="baseline", **parameters):

    allowed = {
        "baseline": set(),
        "randaugment": {"num_ops", "magnitude"},
        "augmix": {"severity", "mixture_width", "chain_depth", "alpha"},
    }
    if augmentation not in allowed or set(parameters) - allowed[augmentation]:
        raise ValueError(f"Invalid augmentation parameters: {augmentation}, {parameters}")

    if augmentation == "baseline":

        return transforms.Compose([
            transforms.RandomCrop(
                32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR10_MEAN,
                CIFAR10_STD,
            ),
        ])

    elif augmentation == "randaugment":

        return transforms.Compose([
            transforms.RandomCrop(
                32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),

            transforms.RandAugment(
                num_ops=parameters.get("num_ops", 2),
                magnitude=parameters.get("magnitude", 9),
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                CIFAR10_MEAN,
                CIFAR10_STD,
            ),
        ])
    elif augmentation == "augmix":

        return transforms.Compose([
            transforms.RandomCrop(
                32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),

            transforms.AugMix(
                severity=parameters.get("severity", 3),
                mixture_width=parameters.get("mixture_width", 3),
                chain_depth=parameters.get("chain_depth", -1),
                alpha=parameters.get("alpha", 1.0),
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                CIFAR10_MEAN,
                CIFAR10_STD,
            ),
        ])


    else:
        raise ValueError(
            f"Unknown augmentation: {augmentation}"
        )


def get_test_transform():

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])


def get_cifar10_dataloaders(
    data_dir="./data",
    batch_size=128,
    num_workers=4,
    augmentation="baseline",
):

    train_transform = get_train_transform(
        augmentation
    )

    test_transform = get_test_transform()

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader
