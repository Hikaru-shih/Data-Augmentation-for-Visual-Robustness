from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.cifar10c import CIFAR10C
from src.models.resnet import get_resnet18
from src.evaluation.evaluator import evaluate


def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = get_resnet18(num_classes=10).to(device)

    checkpoint_path = Path(
        "checkpoints/baseline_resnet18/best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Loaded checkpoint:")
    print("  Epoch:", checkpoint["epoch"])
    print(
        "  Clean test accuracy:",
        f'{checkpoint["test_accuracy"]:.2f}%'
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Gaussian Noise Evaluation
    # --------------------------------------------------

    print()
    print("Gaussian Noise Robustness")
    print("-" * 40)

    for severity in range(1, 6):

        dataset = CIFAR10C(
            root="./data/CIFAR-10-C",
            corruption="gaussian_noise",
            severity=severity,
        )

        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        loss, accuracy = evaluate(
            model,
            loader,
            criterion,
            device,
        )

        print(
            f"Severity {severity}: "
            f"Loss = {loss:.4f} | "
            f"Accuracy = {accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()