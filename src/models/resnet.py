import torch.nn as nn
from torchvision.models import resnet18


def get_resnet18(num_classes=10):
    """
    Create a ResNet-18 adapted for CIFAR-10.

    Main differences from the standard ImageNet ResNet-18:
    1. 7x7 stride-2 convolution -> 3x3 stride-1 convolution
    2. Remove the initial max pooling layer
    3. Output 10 classes
    """

    model = resnet18(weights=None)

    # Adapt the first convolution for 32x32 CIFAR images
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Remove initial max pooling
    model.maxpool = nn.Identity()

    # CIFAR-10 has 10 classes
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model