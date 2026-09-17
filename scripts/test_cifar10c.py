from torch.utils.data import DataLoader

from src.datasets.cifar10c import CIFAR10C


def main():

    dataset = CIFAR10C(
        root="./data/CIFAR-10-C",
        corruption="gaussian_noise",
        severity=1,
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )

    images, labels = next(iter(loader))

    print("Dataset size:", len(dataset))
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Image range:")
    print("  min:", images.min().item())
    print("  max:", images.max().item())


if __name__ == "__main__":
    main()