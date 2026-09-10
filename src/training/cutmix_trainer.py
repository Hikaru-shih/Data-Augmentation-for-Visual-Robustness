from tqdm import tqdm
import numpy as np

from src.augmentations.cutmix import (
    cutmix_data,
    cutmix_loss,
)


def train_one_epoch_cutmix(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    alpha=1.0,
    probability=1.0,
    stats=None,
):
    if not 0 <= probability <= 1:
        raise ValueError("CutMix probability must be in [0, 1]")
    if stats is not None:
        stats.update(batches=0, applied_batches=0)
    model.train()

    total_loss = 0.0
    correct = 0.0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training CutMix",
        leave=False,
    )

    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # Avoid an extra RNG draw at p=1 to preserve the legacy random sequence.
        applied = probability == 1 or (probability > 0 and np.random.random() < probability)
        if applied:
            mixed_images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=alpha)
        else:
            mixed_images, labels_a, labels_b, lam = images, labels, labels, 1.0
        if stats is not None:
            stats["batches"] += 1
            stats["applied_batches"] += int(applied)

        optimizer.zero_grad()

        outputs = model(mixed_images)

        loss = cutmix_loss(
            criterion,
            outputs,
            labels_a,
            labels_b,
            lam,
        ) if applied else criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        # Loss
        total_loss += (
            loss.item()
            * images.size(0)
        )

        # Approximate CutMix training accuracy
        _, predicted = outputs.max(1)

        correct += (
            lam
            * predicted.eq(
                labels_a
            ).sum().item()
            +
            (1.0 - lam)
            * predicted.eq(
                labels_b
            ).sum().item()
        )

        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%"
        )

    epoch_loss = (
        total_loss / total
    )

    epoch_accuracy = (
        100.0 * correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )
