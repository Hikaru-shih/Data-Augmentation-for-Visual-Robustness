"""Analyze saved runs only; all error bars are sample SD across training seeds."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.summarize_multiseed import collect, run_keys, METHODS

# Explicit grouping used for this analysis; each corruption receives equal weight.
CATEGORIES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
LABELS = dict(zip(METHODS, ["Baseline", "Mixup", "CutMix", "RandAugment", "AugMix Transform"]))


def validate_detail(frame):
    expected = {(c, s) for names in CATEGORIES.values() for c in names for s in range(1, 6)}
    if len(frame) != 75 or set(zip(frame.corruption, frame.severity)) != expected:
        raise ValueError("Expected exactly 15 corruptions x 5 severities without duplicates")
    if not np.isfinite(frame.accuracy).all() or not frame.accuracy.between(0, 100).all():
        raise ValueError("Invalid accuracy values")


def summarize(frame, dimensions):
    # First average within each seed, then compute variability across seeds.
    per_seed = frame.groupby(["method", "seed"] + dimensions, as_index=False).accuracy.mean()
    baseline = per_seed[per_seed.method == "baseline"].drop(columns="method").rename(columns={"accuracy": "baseline"})
    paired = per_seed.merge(baseline, on=["seed"] + dimensions, validate="many_to_one")
    paired["delta"] = paired.accuracy - paired.baseline
    stats = paired.groupby(["method"] + dimensions, as_index=False).agg(
        seeds=("seed", "nunique"), accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"), delta_mean=("delta", "mean"),
        delta_std=("delta", "std"), delta_min=("delta", "min"), delta_max=("delta", "max"))
    return paired, stats


def main():
    overall = collect()  # Checks histories, splits, configs, checkpoints and CSV hashes.
    frames = []
    mapping = {name: category for category, names in CATEGORIES.items() for name in names}
    for method, seed, key in run_keys():
        detail = pd.read_csv(Path("results") / key / "robustness/corruption_results.csv")
        validate_detail(detail)
        saved = pd.read_csv(Path("results") / key / "robustness/corruption_summary.csv").set_index("corruption")
        means = detail.groupby("corruption").accuracy.mean()
        if not np.allclose(means, saved.loc[means.index, "mean_accuracy"], rtol=0, atol=1e-6):
            raise ValueError(f"Detail/summary mismatch: {key}")
        detail["method"], detail["seed"] = method, seed
        detail["category"] = detail.corruption.map(mapping)
        frames.append(detail)
    data = pd.concat(frames, ignore_index=True)
    output = Path("results/analysis/multiseed-42-43-44")
    output.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, dimensions in {"corruption": ["corruption"], "category": ["category"],
                             "severity": ["severity"], "category_severity": ["category", "severity"]}.items():
        paired, stats = summarize(data, dimensions)
        paired.to_csv(output / f"{name}_per_seed.csv", index=False)
        stats.to_csv(output / f"{name}_summary.csv", index=False)
        tables[name] = stats

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["clean_accuracy", "mca"], ["Clean accuracy", "Mean corruption accuracy"]):
        grouped = overall.groupby("method")[metric].agg(["mean", "std"]).loc[METHODS]
        ax.errorbar(range(5), grouped["mean"], yerr=grouped["std"], fmt="o", capsize=4)
        ax.set_xticks(range(5), [LABELS[m] for m in METHODS], rotation=25, ha="right")
        ax.set(title=title, ylabel="Accuracy (%)")
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Three seeds: mean +/- sample SD")
    fig.tight_layout()
    fig.savefig(output / "overall.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, method in zip(axes.flat, METHODS[1:]):
        rows = tables["corruption"].query("method == @method").sort_values("delta_mean")
        ax.errorbar(rows.delta_mean, range(15), xerr=rows.delta_std, fmt="o", capsize=3)
        ax.set_yticks(range(15), rows.corruption)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set(title=LABELS[method], xlabel="Paired delta vs baseline (pp), mean +/- SD")
        ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "corruption_deltas.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, category in zip(axes.flat, CATEGORIES):
        for method in METHODS:
            rows = tables["category_severity"].query("category == @category and method == @method").sort_values("severity")
            ax.errorbar(rows.severity, rows.accuracy_mean, yerr=rows.accuracy_std, marker="o", capsize=3, label=LABELS[method])
        ax.set(title=category.title(), xlabel="Severity", ylabel="Accuracy (%)", xticks=range(1, 6))
        ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Category severity curves: mean +/- sample SD across seeds")
    fig.tight_layout()
    fig.savefig(output / "category_severity.png", dpi=180)
    plt.close(fig)
    report = ["# Corruption analysis", "", "Seeds: 42, 43, 44. SD uses ddof=1, not standard error or a confidence interval.",
              "Within each seed, corruptions/severities are equally weighted before across-seed statistics.",
              "Deltas are paired by training seed. Category means are not averaged to calculate overall mCA.", "",
              "## Category mapping", ""]
    report.extend(f"- {category}: {', '.join(names)}" for category, names in CATEGORIES.items())
    for method in ["cutmix", "augmix"]:
        rows = tables["corruption"].query("method == @method").sort_values("delta_mean", ascending=method == "cutmix").head(5)
        report += ["", f"## {LABELS[method]}: five {'lowest' if method == 'cutmix' else 'highest'} deltas", ""]
        for row in rows.itertuples():
            report.append(f"- {row.corruption}: {row.delta_mean:+.2f} +/- {row.delta_std:.2f} pp; seed range [{row.delta_min:+.2f}, {row.delta_max:+.2f}].")
    report += ["", "These are descriptive observations from three seeds; they do not establish significance or a causal mechanism."]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Analysis saved to {output}")


if __name__ == "__main__":
    main()
