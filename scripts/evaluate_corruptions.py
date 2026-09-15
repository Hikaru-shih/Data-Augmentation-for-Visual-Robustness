import csv
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.cifar10c import CIFAR10C
from src.models.resnet import get_resnet18
from src.evaluation.evaluator import evaluate
from src.utils.runs import file_hash


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def main():

    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide experiment name.\n"
            "Example:\n"
            "python -m scripts.evaluate_corruptions baseline_resnet18"
        )

    experiment_name = sys.argv[1]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Experiment:", experiment_name)
    print("Device:", device)


    checkpoint_path = (
        Path("checkpoints")
        / experiment_name
        / "best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    checkpoint_digest = file_hash(checkpoint_path)
    config = checkpoint["config"]
    model = get_resnet18(num_classes=config["dataset"]["num_classes"]).to(device)
    dataset_name = config["dataset"]["name"]
    corruption_root = Path(config["dataset"].get("corruption_dir", str(Path(config["dataset"]["data_dir"]) / "CIFAR-10-C")))
    dataset_kwargs = {}
    if dataset_name == "cifar100":
        from src.datasets.cifar100 import normalization
        normalization(config)
        norm = config["dataset"]["normalization"]
        dataset_kwargs = dict(num_classes=100, mean=norm["mean"], std=norm["std"])
        if "corruption_dir" not in config["dataset"]:
            raise ValueError("CIFAR-100 requires an explicit corruption_dir")
    data_hashes = {}
    if dataset_name == "cifar100":
        # Validate identity against official clean-test fine labels before scoring.
        import numpy as np
        from torchvision.datasets import CIFAR100
        clean = CIFAR100(config["dataset"]["data_dir"], train=False, download=False)
        labels = np.load(corruption_root / "labels.npy")
        expected = np.asarray(clean.targets)
        if labels.shape == (50000,):
            expected = np.tile(expected, 5)
        if not np.array_equal(labels, expected):
            raise ValueError("Corruption labels do not match CIFAR-100 fine test labels")
        data_hashes = {name: file_hash(corruption_root / name) for name in ["labels.npy"] + [c + ".npy" for c in CORRUPTIONS]}

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Loaded checkpoint:")
    print("  Epoch:", checkpoint["epoch"])
    print(
        "  Clean test accuracy:",
        f'{checkpoint["test_accuracy"]:.2f}%'
    )

    criterion = nn.CrossEntropyLoss()

    output_dir = (
        Path("results")
        / experiment_name
        / "robustness"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_path = output_dir / "corruption_results.csv"
    summary_path = output_dir / "corruption_summary.csv"

    detailed_results = []
    summary_results = []

    print()
    print(f"{dataset_name.upper()}-C Robustness Evaluation")
    print("=" * 60)

    for corruption in CORRUPTIONS:

        print()
        print(f"[{corruption}]")

        corruption_accuracies = []

        for severity in range(1, 6):

            dataset = CIFAR10C(
                root=corruption_root,
                corruption=corruption,
                severity=severity,
                **dataset_kwargs,
            )

            loader = DataLoader(
                dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                num_workers=config["training"].get("num_workers", 4),
                pin_memory=torch.cuda.is_available(),
            )

            loss, accuracy = evaluate(
                model,
                loader,
                criterion,
                device,
            )

            corruption_accuracies.append(accuracy)

            detailed_results.append({
                "corruption": corruption,
                "severity": severity,
                "loss": loss,
                "accuracy": accuracy,
            })

            print(
                f"  Severity {severity}: "
                f"Loss = {loss:.4f} | "
                f"Accuracy = {accuracy:.2f}%"
            )

        mean_accuracy = (
            sum(corruption_accuracies)
            / len(corruption_accuracies)
        )

        summary_results.append({
            "corruption": corruption,
            "mean_accuracy": mean_accuracy,
        })

        print(
            f"  Mean Accuracy: {mean_accuracy:.2f}%"
        )

    overall_mca = (
        sum(
            item["mean_accuracy"]
            for item in summary_results
        )
        / len(summary_results)
    )

    with open(
        detail_path,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "corruption",
                "severity",
                "loss",
                "accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(detailed_results)

    with open(
        summary_path,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "corruption",
                "mean_accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_results)

        writer.writerow({
            "corruption": "overall_mean",
            "mean_accuracy": overall_mca,
        })

    print()
    print("=" * 60)
    if file_hash(checkpoint_path) != checkpoint_digest:
        raise RuntimeError("Checkpoint changed during evaluation; rerun evaluation")
    (output_dir / "metadata.json").write_text(json.dumps({
        "checkpoint_sha256": checkpoint_digest,
        "dataset": dataset_name,
        "data_sha256": data_hashes,
        "epoch": checkpoint["epoch"],
        "protocol": config.get("run", {}).get("protocol", "legacy-test-selected"),
        "files": {path.name: file_hash(path) for path in (detail_path, summary_path)},
    }, indent=2), encoding="utf-8")
    print("Evaluation completed.")
    print(
        f"Overall Mean Corruption Accuracy: "
        f"{overall_mca:.2f}%"
    )
    print("Detailed results saved to:")
    print(detail_path)
    print("Summary saved to:")
    print(summary_path)


if __name__ == "__main__":
    main()
