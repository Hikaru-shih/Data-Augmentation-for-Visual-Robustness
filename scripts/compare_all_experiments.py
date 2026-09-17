from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
from src.utils.runs import verify_evaluation


EXPERIMENTS = {
    "Baseline": "baseline_resnet18",
    "Mixup": "mixup_resnet18",
    "CutMix": "cutmix_resnet18",
    "RandAugment": "randaugment_resnet18",
    "AugMix Transform": "augmix_resnet18",
}


def load_clean_accuracy(experiment_name):
    checkpoint_path = (
        Path("checkpoints")
        / experiment_name
        / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    return checkpoint["test_accuracy"]


def load_corruption_summary(experiment_name):
    verify_evaluation(experiment_name)
    summary_path = (
        Path("results")
        / experiment_name
        / "robustness"
        / "corruption_summary.csv"
    )

    return pd.read_csv(summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run keys printed by training; labels are the run keys")
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    experiments = {key: key for key in args.runs}
    if len(experiments) != len(args.runs):
        raise ValueError("Duplicate runs")

    output_dir = (
        Path("results")
        / "comparisons"
        / args.output_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Overall comparison
    # --------------------------------------------------

    overall_rows = []

    corruption_tables = []

    for display_name, experiment_name in experiments.items():

        clean_accuracy = load_clean_accuracy(
            experiment_name
        )

        summary = load_corruption_summary(
            experiment_name
        )

        mca = summary.loc[
            summary["corruption"] == "overall_mean",
            "mean_accuracy",
        ].iloc[0]

        overall_rows.append({
            "method": display_name,
            "clean_accuracy": clean_accuracy,
            "mca": mca,
        })

        corruption_only = summary[
            summary["corruption"] != "overall_mean"
        ].copy()

        corruption_only = corruption_only.rename(
            columns={
                "mean_accuracy": display_name
            }
        )

        corruption_tables.append(
            corruption_only
        )

    overall_df = pd.DataFrame(
        overall_rows
    )

    overall_csv = (
        output_dir
        / "overall_comparison.csv"
    )

    overall_df.to_csv(
        overall_csv,
        index=False,
    )

    # --------------------------------------------------
    # Merge all corruption results
    # --------------------------------------------------

    corruption_df = corruption_tables[0]

    for table in corruption_tables[1:]:

        corruption_df = corruption_df.merge(
            table,
            on="corruption",
        )

    corruption_csv = (
        output_dir
        / "corruption_comparison.csv"
    )

    corruption_df.to_csv(
        corruption_csv,
        index=False,
    )

    # --------------------------------------------------
    # Plot 1: mCA comparison
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.bar(
        overall_df["method"],
        overall_df["mca"],
    )

    plt.ylabel("Mean Corruption Accuracy (%)")
    plt.xlabel("Method")
    plt.title(
        "Corruption Robustness Comparison"
    )

    plt.ylim(
        max(0, overall_df["mca"].min() - 5),
        min(100, overall_df["mca"].max() + 5),
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    mca_plot = (
        output_dir
        / "mca_comparison.png"
    )

    plt.savefig(
        mca_plot,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Plot 2: Clean accuracy comparison
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.bar(
        overall_df["method"],
        overall_df["clean_accuracy"],
    )

    plt.ylabel("Clean Accuracy (%)")
    plt.xlabel("Method")
    plt.title(
        "Clean Accuracy Comparison"
    )

    plt.ylim(
        max(
            0,
            overall_df["clean_accuracy"].min() - 2,
        ),
        min(
            100,
            overall_df["clean_accuracy"].max() + 2,
        ),
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    clean_plot = (
        output_dir
        / "clean_accuracy_comparison.png"
    )

    plt.savefig(
        clean_plot,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Plot 3: Clean vs mCA
    # --------------------------------------------------

    plt.figure(figsize=(7, 6))

    for _, row in overall_df.iterrows():

        plt.scatter(
            row["clean_accuracy"],
            row["mca"],
            s=80,
        )

        plt.annotate(
            row["method"],
            (
                row["clean_accuracy"],
                row["mca"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Clean Accuracy (%)")
    plt.ylabel("Mean Corruption Accuracy (%)")

    plt.title(
        "Clean Accuracy vs Corruption Robustness"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    scatter_plot = (
        output_dir
        / "clean_vs_mca.png"
    )

    plt.savefig(
        scatter_plot,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Plot 4: All corruptions
    # --------------------------------------------------

    plot_df = corruption_df.set_index(
        "corruption"
    )

    plot_df.plot(
        kind="bar",
        figsize=(15, 7),
    )

    plt.ylabel("Mean Accuracy (%)")
    plt.xlabel("Corruption")
    plt.title(
        "Robustness Across Corruption Types"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend(
        title="Method",
    )

    plt.tight_layout()

    corruption_plot = (
        output_dir
        / "corruption_comparison.png"
    )

    plt.savefig(
        corruption_plot,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("Overall Comparison")
    print("=" * 70)

    print(
        overall_df
        .sort_values(
            "mca",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("=" * 70)
    print("Best Method Per Corruption")
    print("=" * 70)

    method_columns = list(
        experiments.keys()
    )

    for _, row in corruption_df.iterrows():

        best_method = row[
            method_columns
        ].idxmax()

        best_accuracy = row[
            best_method
        ]

        print(
            f"{row['corruption']:20s} "
            f"{best_method:12s} "
            f"{best_accuracy:.2f}%"
        )

    print()
    print("Saved:")
    print(" -", overall_csv)
    print(" -", corruption_csv)
    print(" -", mca_plot)
    print(" -", clean_plot)
    print(" -", scatter_plot)
    print(" -", corruption_plot)


if __name__ == "__main__":
    main()
