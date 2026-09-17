import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from src.utils.runs import verify_evaluation


def load_summary(experiment_name):
    verify_evaluation(experiment_name)
    summary_path = (
        Path("results")
        / experiment_name
        / "robustness"
        / "corruption_summary.csv"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}"
        )

    df = pd.read_csv(summary_path)

    return df


def load_checkpoint_clean_accuracy(experiment_name):
    import torch

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
        map_location="cpu",
    )

    return checkpoint["test_accuracy"]


def main():

    if len(sys.argv) < 3:
        raise ValueError(
            "Please provide two experiment names.\n"
            "Example:\n"
            "python -m scripts.compare_experiments "
            "baseline_resnet18 mixup_resnet18"
        )

    experiment_a = sys.argv[1]
    experiment_b = sys.argv[2]

    print("Comparing:")
    print(" A:", experiment_a)
    print(" B:", experiment_b)
    print()

    # --------------------------------------------------
    # Load corruption summaries
    # --------------------------------------------------

    summary_a = load_summary(experiment_a)
    summary_b = load_summary(experiment_b)

    # Separate overall mCA
    overall_a = summary_a[
        summary_a["corruption"] == "overall_mean"
    ]["mean_accuracy"].iloc[0]

    overall_b = summary_b[
        summary_b["corruption"] == "overall_mean"
    ]["mean_accuracy"].iloc[0]

    corruption_a = summary_a[
        summary_a["corruption"] != "overall_mean"
    ].copy()

    corruption_b = summary_b[
        summary_b["corruption"] != "overall_mean"
    ].copy()

    # --------------------------------------------------
    # Merge corruption results
    # --------------------------------------------------

    comparison = corruption_a.merge(
        corruption_b,
        on="corruption",
        suffixes=("_a", "_b"),
    )

    comparison["delta"] = (
        comparison["mean_accuracy_b"]
        - comparison["mean_accuracy_a"]
    )

    comparison = comparison.sort_values(
        "delta",
        ascending=False,
    )

    # --------------------------------------------------
    # Clean accuracy
    # --------------------------------------------------

    clean_a = load_checkpoint_clean_accuracy(
        experiment_a
    )

    clean_b = load_checkpoint_clean_accuracy(
        experiment_b
    )

    clean_delta = clean_b - clean_a
    mca_delta = overall_b - overall_a

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    output_dir = (
        Path("results")
        / "comparisons"
        / f"{experiment_a}_vs_{experiment_b}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    csv_path = (
        output_dir
        / "corruption_comparison.csv"
    )

    comparison.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------
    # Save overall summary
    # --------------------------------------------------

    overall_df = pd.DataFrame({
        "experiment": [
            experiment_a,
            experiment_b,
        ],
        "clean_accuracy": [
            clean_a,
            clean_b,
        ],
        "mca": [
            overall_a,
            overall_b,
        ],
    })

    overall_path = (
        output_dir
        / "overall_comparison.csv"
    )

    overall_df.to_csv(
        overall_path,
        index=False,
    )

    # --------------------------------------------------
    # Plot 1:
    # corruption accuracy comparison
    # --------------------------------------------------

    plot_data = comparison.sort_values(
        "mean_accuracy_a",
        ascending=True,
    )

    y_positions = range(len(plot_data))

    bar_height = 0.4

    plt.figure(figsize=(11, 8))

    plt.barh(
        [y - bar_height / 2 for y in y_positions],
        plot_data["mean_accuracy_a"],
        height=bar_height,
        label=experiment_a,
    )

    plt.barh(
        [y + bar_height / 2 for y in y_positions],
        plot_data["mean_accuracy_b"],
        height=bar_height,
        label=experiment_b,
    )

    plt.yticks(
        list(y_positions),
        plot_data["corruption"],
    )

    plt.xlabel("Mean Accuracy (%)")
    plt.ylabel("Corruption")
    plt.title(
        f"{experiment_a} vs {experiment_b}"
    )

    plt.legend()
    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    comparison_plot_path = (
        output_dir
        / "corruption_accuracy_comparison.png"
    )

    plt.savefig(
        comparison_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Plot 2:
    # Delta per corruption
    # --------------------------------------------------

    delta_plot = comparison.sort_values(
        "delta",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        delta_plot["corruption"],
        delta_plot["delta"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        f"Accuracy Difference: "
        f"{experiment_b} - {experiment_a} "
        f"(percentage points)"
    )

    plt.ylabel("Corruption")

    plt.title(
        f"Robustness Change: "
        f"{experiment_a} → {experiment_b}"
    )

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    delta_plot_path = (
        output_dir
        / "corruption_delta.png"
    )

    plt.savefig(
        delta_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Terminal summary
    # --------------------------------------------------

    print("=" * 60)
    print("Overall Results")
    print("=" * 60)

    print(
        f"{experiment_a}: "
        f"Clean = {clean_a:.2f}% | "
        f"mCA = {overall_a:.2f}%"
    )

    print(
        f"{experiment_b}: "
        f"Clean = {clean_b:.2f}% | "
        f"mCA = {overall_b:.2f}%"
    )

    print()

    print(
        f"Clean Accuracy Change: "
        f"{clean_delta:+.2f} pp"
    )

    print(
        f"mCA Change: "
        f"{mca_delta:+.2f} pp"
    )

    print()
    print("=" * 60)
    print("Largest Robustness Improvements")
    print("=" * 60)

    print(
        comparison[
            [
                "corruption",
                "mean_accuracy_a",
                "mean_accuracy_b",
                "delta",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print()
    print("=" * 60)
    print("Largest Robustness Regressions")
    print("=" * 60)

    print(
        comparison[
            [
                "corruption",
                "mean_accuracy_a",
                "mean_accuracy_b",
                "delta",
            ]
        ]
        .tail(5)
        .sort_values("delta")
        .to_string(index=False)
    )

    print()
    print("Saved:")
    print(" -", csv_path)
    print(" -", overall_path)
    print(" -", comparison_plot_path)
    print(" -", delta_plot_path)


if __name__ == "__main__":
    main()
