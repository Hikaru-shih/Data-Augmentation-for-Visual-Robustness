"""Per-corruption paired analysis; called only after run provenance validation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {"baseline": "Baseline", "cutmix_p1": "CutMix p=1", "cutmix_p05": "CutMix p=0.5"}


def paired_statistics(data, dimensions):
    per_seed = data.groupby(["method", "seed"] + dimensions, as_index=False).accuracy.mean()
    accuracy = per_seed.groupby(["method"] + dimensions, as_index=False).agg(
        mean=("accuracy", "mean"), std=("accuracy", "std"), seeds=("seed", "nunique"))
    target = per_seed[per_seed.method == "cutmix_p05"].drop(columns="method")
    paired = []
    for control in ("baseline", "cutmix_p1"):
        other = per_seed[per_seed.method == control].drop(columns="method")
        merged = target.merge(other, on=["seed"] + dimensions, how="outer",
                              suffixes=("_p05", "_control"), validate="one_to_one", indicator=True)
        if not merged["_merge"].eq("both").all():
            raise ValueError("Missing paired seed or condition")
        merged["control"] = control
        merged["delta"] = merged.accuracy_p05 - merged.accuracy_control
        paired.append(merged.drop(columns="_merge"))
    paired = pd.concat(paired, ignore_index=True)
    summary = paired.groupby(["control"] + dimensions, as_index=False).agg(
        mean=("delta", "mean"), std=("delta", "std"), minimum=("delta", "min"),
        maximum=("delta", "max"), seeds=("seed", "nunique"))
    if not summary.seeds.eq(3).all() or not accuracy.seeds.eq(3).all():
        raise ValueError("Expected three seeds per condition")
    return per_seed, accuracy, paired, summary


def analyze(data, output):
    tables = {}
    for name, dims in (("corruption", ["corruption"]), ("corruption_severity", ["corruption", "severity"])):
        tables[name] = paired_statistics(data, dims)
        for suffix, table in zip(("per_seed", "accuracy", "paired", "delta"), tables[name]):
            table.to_csv(output / f"{name}_{suffix}.csv", index=False)
    stats = tables["corruption"][3]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True)
    for ax, control in zip(axes, ("baseline", "cutmix_p1")):
        frame = stats[stats.control == control].sort_values("mean")
        ax.errorbar(frame["mean"], range(len(frame)), xerr=frame["std"], fmt="o", capsize=3)
        ax.set_yticks(range(len(frame)), frame.corruption)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set(title=f"p=0.5 minus {LABELS[control]}", xlabel="Paired accuracy delta (pp), mean +/- sample SD")
        ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "corruption_deltas.png", dpi=180)
    plt.close(fig)

    accuracy = tables["corruption_severity"][1]
    corruptions = list(data.corruption.unique())
    for filename, names, shape in (("noise_severity.png", ["gaussian_noise", "shot_noise", "impulse_noise"], (1, 3)),
                                   ("all_severity.png", corruptions, (5, 3))):
        fig, axes = plt.subplots(*shape, figsize=(15, shape[0] * 3.6), squeeze=False, sharey=True)
        for ax, corruption in zip(axes.flat, names):
            for method, label in LABELS.items():
                frame = accuracy[(accuracy.method == method) & (accuracy.corruption == corruption)].sort_values("severity")
                ax.errorbar(frame.severity, frame["mean"], yerr=frame["std"], marker="o", capsize=3, label=label)
            ax.set(title=corruption, xlabel="Severity", ylabel="Accuracy (%)", xticks=range(1, 6), ylim=(0, 100))
            ax.grid(alpha=.25)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle("Three seeds: mean +/- sample SD (not confidence intervals)")
        fig.tight_layout(rect=(0, 0, 1, .96 if shape[0] == 1 else .98))
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    report = ["# CutMix probability: detailed analysis", "", "Accuracy and paired delta SD are computed across seeds 42,43,44 (ddof=1).",
              "Severity is averaged within each seed for corruption-level tables. All 15 corruptions are included.",
              "The predeclared primary noise endpoint includes Gaussian and shot only; impulse is secondary.", ""]
    for control in ("cutmix_p1", "baseline"):
        frame = stats[stats.control == control].sort_values("mean", ascending=False)
        report += [f"## p=0.5 minus {LABELS[control]}", "",
                   f"Positive mean: {int((frame['mean'] > 0).sum())}/15; positive in all seeds: {int((frame.minimum > 0).sum())}/15; negative in all seeds: {int((frame.maximum < 0).sum())}/15.", "",
                   "| Corruption | Mean delta (pp) | Sample SD | Seed min | Seed max |", "|---|---:|---:|---:|---:|"]
        for row in frame.itertuples():
            report.append(f"| {row.corruption} | {row.mean:+.2f} | {row.std:.2f} | {row.minimum:+.2f} | {row.maximum:+.2f} |")
        report.append("")
    severity = tables["corruption_severity"][3]
    report += ["## Noise severity: p=0.5 minus p=1", "", "| Corruption | Severity | Mean delta (pp) | SD |", "|---|---:|---:|---:|"]
    for row in severity[(severity.control == "cutmix_p1") & severity.corruption.isin(["gaussian_noise", "shot_noise", "impulse_noise"])].itertuples():
        report.append(f"| {row.corruption} | {row.severity} | {row.mean:+.2f} | {row.std:.2f} |")
    report += ["", "Descriptive exploratory comparisons; neither statistical significance nor a causal mechanism is established."]
    (output / "detail_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
