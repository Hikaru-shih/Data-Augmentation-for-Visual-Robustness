"""Regenerate validated per-dataset analysis, then compare within-task effects."""
import subprocess
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    directories = {"CIFAR-10": "cutmix-probability", "CIFAR-100": "cifar100_cutmix-probability"}
    effects, corruptions, severities = [], [], []
    for dataset, folder in directories.items():
        subprocess.run([sys.executable, "-m", "scripts.compare_cutmix_probability", "--dataset", dataset.lower().replace("-", "")], check=True)
        directory = Path("results/analysis") / folder
        paired = pd.read_csv(directory / "paired_deltas.csv")
        for control, group in paired.groupby("control"):
            for metric in ("clean", "mca", "noise"):
                effects.append(dict(dataset=dataset, control=control, metric=metric,
                                    mean=group[metric].mean(), std=group[metric].std(ddof=1),
                                    minimum=group[metric].min(), maximum=group[metric].max()))
        corruptions.append(pd.read_csv(directory / "corruption_delta.csv").assign(dataset=dataset))
        severities.append(pd.read_csv(directory / "corruption_severity_delta.csv").assign(dataset=dataset))
    output = Path("results/analysis/cross_dataset")
    output.mkdir(parents=True, exist_ok=True)
    effect = pd.DataFrame(effects)
    corruption = pd.concat(corruptions, ignore_index=True)
    severity = pd.concat(severities, ignore_index=True)
    for name, table in (("paired_effects", effect), ("corruption_effects", corruption), ("severity_effects", severity)):
        table.to_csv(output / f"{name}.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ("clean", "mca", "noise")):
        rows = effect[(effect.control == "cutmix_p1") & (effect.metric == metric)]
        ax.errorbar(range(2), rows["mean"], yerr=rows["std"], fmt="o", capsize=5)
        ax.set_xticks(range(2), rows.dataset)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(title=metric, ylabel="p=0.5 minus p=1 (pp)")
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Within-dataset paired effects: mean +/- sample SD, three seeds")
    fig.tight_layout()
    fig.savefig(output / "paired_effects.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, noise in zip(axes, ("gaussian_noise", "shot_noise", "impulse_noise")):
        for dataset in directories:
            rows = severity[(severity.dataset == dataset) & (severity.control == "cutmix_p1") & (severity.corruption == noise)].sort_values("severity")
            ax.errorbar(rows.severity, rows["mean"], yerr=rows["std"], marker="o", capsize=3, label=dataset)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(title=noise, xlabel="Severity", ylabel="p=0.5 minus p=1 (pp)", xticks=range(1, 6))
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle("Paired effects by severity: mean +/- sample SD")
    fig.tight_layout()
    fig.savefig(output / "noise_effects.png", dpi=180)
    plt.close(fig)
    pivot = corruption[corruption.control == "cutmix_p1"].pivot(index="corruption", columns="dataset", values="minimum")
    common = pivot[(pivot > 0).all(axis=1)].index.tolist()
    report = ["# Cross-dataset summary", "", "Effects are paired within each dataset/seed; datasets are not pooled.",
              "Noise endpoint = Gaussian and shot, averaged over five severities. SD is sample SD, not CI.", "",
              "Positive p=0.5 minus p=1 corruption effects in all three seeds of both datasets: " + ", ".join(common), "",
              "| Dataset | Control | Metric | Mean delta (pp) | SD |", "|---|---|---|---:|---:|"]
    for row in effect.itertuples():
        report.append(f"| {row.dataset} | {row.control} | {row.metric} | {row.mean:+.2f} | {row.std:.2f} |")
    report += ["", "This is limited cross-task evidence within the CIFAR family and one architecture. No formal significance, equivalence or causal mechanism claim."]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved {output}; shared consistent improvements: {common}")


if __name__ == "__main__":
    main()
