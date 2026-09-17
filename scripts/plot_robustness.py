from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import sys


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide experiment name.\n"
            "Example:\n"
            "python -m scripts.plot_robustness baseline_resnet18"
        )

    experiment_name = sys.argv[1]

    robustness_dir = (
        Path("results")
        / experiment_name
        / "robustness"
    )

    summary_path = robustness_dir / "corruption_summary.csv"
    detail_path = robustness_dir / "corruption_results.csv"

    figure_dir = robustness_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load results
    # --------------------------------------------------

    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)

    # Remove overall_mean row from bar chart
    summary_bar = summary[
        summary["corruption"] != "overall_mean"
    ].copy()

    # Sort for easier comparison
    summary_bar = summary_bar.sort_values(
        "mean_accuracy",
        ascending=True,
    )

    # --------------------------------------------------
    # Plot 1: Mean accuracy by corruption
    # --------------------------------------------------

    plt.figure(figsize=(10, 7))

    plt.barh(
        summary_bar["corruption"],
        summary_bar["mean_accuracy"],
    )

    plt.xlabel("Mean Accuracy (%)")
    plt.ylabel("Corruption")
    plt.title(
        f"{experiment_name}: Corruption Robustness"
    )

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    bar_path = (
        figure_dir
        / "corruption_mean_accuracy.png"
    )

    plt.savefig(
        bar_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Plot 2: Accuracy vs severity
    # --------------------------------------------------

    plt.figure(figsize=(11, 7))

    for corruption in detail["corruption"].unique():

        corruption_data = detail[
            detail["corruption"] == corruption
        ]

        plt.plot(
            corruption_data["severity"],
            corruption_data["accuracy"],
            marker="o",
            label=corruption,
        )

    plt.xlabel("Severity")
    plt.ylabel("Accuracy (%)")
    plt.title(
        "Accuracy Degradation Across Corruption Severity"
    )

    plt.xticks([1, 2, 3, 4, 5])

    plt.grid(alpha=0.3)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )

    plt.tight_layout()

    severity_path = (
        figure_dir
        / "severity_curves_all.png"
    )

    plt.savefig(
        severity_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Plot 3: Noise corruptions only
    # --------------------------------------------------

    noise_corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
    ]

    plt.figure(figsize=(8, 5))

    for corruption in noise_corruptions:

        corruption_data = detail[
            detail["corruption"] == corruption
        ]

        plt.plot(
            corruption_data["severity"],
            corruption_data["accuracy"],
            marker="o",
            label=corruption,
        )

    plt.xlabel("Severity")
    plt.ylabel("Accuracy (%)")
    plt.title(
        f"{experiment_name}: Robustness to Noise Corruptions"
    )

    plt.xticks([1, 2, 3, 4, 5])

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    noise_path = (
        figure_dir
        / "noise_severity_curves.png"
    )

    plt.savefig(
        noise_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("Robustness plots saved:")
    print(" -", bar_path)
    print(" -", severity_path)
    print(" -", noise_path)


if __name__ == "__main__":
    main()
