from tqdm import tqdm

from src.augmentations.mixup import (
    mixup_data,
    mixup_loss,
)


def train_one_epoch_mixup(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    alpha=1.0,
):
    model.train()

    total_loss = 0.0
    correct = 0.0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training Mixup",
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

        mixed_images, labels_a, labels_b, lam = mixup_data(
            images,
            labels,
            alpha=alpha,
        )

        optimizer.zero_grad()

        outputs = model(mixed_images)

        loss = mixup_loss(
            criterion,
            outputs,
            labels_a,
            labels_b,
            lam,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        _, predicted = outputs.max(1)

        correct += (
            lam
            * predicted.eq(labels_a).sum().item()
            + (1.0 - lam)
            * predicted.eq(labels_b).sum().item()
        )

        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%"
        )

    epoch_loss = total_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy