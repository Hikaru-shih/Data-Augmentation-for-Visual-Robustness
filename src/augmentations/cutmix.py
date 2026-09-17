import numpy as np
import torch


def rand_bbox(size, lam):
    """
    Generate a random bounding box for CutMix.

    size:
        [batch_size, channels, height, width]
    """

    height = size[2]
    width = size[3]

    cut_ratio = np.sqrt(1.0 - lam)

    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)

    # Random center
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(
        cx - cut_width // 2,
        0,
        width,
    )

    x2 = np.clip(
        cx + cut_width // 2,
        0,
        width,
    )

    y1 = np.clip(
        cy - cut_height // 2,
        0,
        height,
    )

    y2 = np.clip(
        cy + cut_height // 2,
        0,
        height,
    )

    return x1, y1, x2, y2


def cutmix_data(images, labels, alpha=1.0):

    if alpha <= 0:
        return images, labels, labels, 1.0

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)

    batch_size = images.size(0)

    # Shuffle the batch
    index = torch.randperm(
        batch_size,
        device=images.device,
    )

    labels_a = labels
    labels_b = labels[index]

    # Generate CutMix region
    x1, y1, x2, y2 = rand_bbox(
        images.size(),
        lam,
    )

    # Important:
    # clone so we do not modify the original tensor
    mixed_images = images.clone()

    mixed_images[
        :,
        :,
        y1:y2,
        x1:x2
    ] = images[
        index,
        :,
        y1:y2,
        x1:x2
    ]

    # Recalculate lambda based on the actual
    # clipped bounding-box area
    box_area = (x2 - x1) * (y2 - y1)

    image_area = (
        images.size(2)
        * images.size(3)
    )

    lam = 1.0 - box_area / image_area

    return (
        mixed_images,
        labels_a,
        labels_b,
        lam,
    )


def cutmix_loss(
    criterion,
    predictions,
    labels_a,
    labels_b,
    lam,
):

    return (
        lam * criterion(
            predictions,
            labels_a,
        )
        + (1.0 - lam)
        * criterion(
            predictions,
            labels_b,
        )
    )