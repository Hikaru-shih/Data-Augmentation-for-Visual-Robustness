import torch
import numpy as np


def mixup_data(images, labels, alpha=1.0):
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)

    batch_size = images.size(0)

    index = torch.randperm(batch_size, device=images.device)

    mixed_images = (
        lam * images
        + (1.0 - lam) * images[index]
    )

    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def mixup_loss(
    criterion,
    predictions,
    labels_a,
    labels_b,
    lam,
):
    return (
        lam * criterion(predictions, labels_a)
        + (1.0 - lam) * criterion(predictions, labels_b)
    )   