from src.datasets.cifar10 import get_cifar10_dataloaders


def main():
    train_loader, test_loader = get_cifar10_dataloaders()

    images, labels = next(iter(train_loader))

    print("Train batches:", len(train_loader))
    print("Test batches:", len(test_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    print("Min pixel value:", images.min().item())
    print("Max pixel value:", images.max().item())


if __name__ == "__main__":
    main()