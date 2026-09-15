"""Compare the predeclared p=0.5 ablation to existing p=1 and baseline runs."""
import json
import argparse
from pathlib import Path
import pandas as pd
import torch
import yaml
from scripts.summarize_multiseed import normalized
from scripts.analyze_corruptions import validate_detail
from src.utils.runs import verify_evaluation, file_hash
from scripts.cutmix_detail_analysis import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    dataset = parser.parse_args().dataset
    prefix = "cifar100_" if dataset == "cifar100" else ""
    protocol = "cifar100-transfer-v1" if dataset == "cifar100" else "validation-v1"
    data_identity = None
    rows, reference, split = [], None, None
    details = []
    for seed in (42, 43, 44):
        for method, experiment in [("baseline", "baseline_resnet18"), ("cutmix_p1", "cutmix_resnet18"), ("cutmix_p05", "cutmix_p05_resnet18")]:
            key = f"{prefix}{experiment}/seed_{seed}/validation-v1"
            directory = Path("results") / key
            verify_evaluation(key)
            config = yaml.safe_load((directory / "config.yaml").read_text())
            expected = 0.5 if method == "cutmix_p05" else 1.0
            if config["seed"] != seed or config["run"]["protocol"] != protocol or config["dataset"]["name"] != dataset:
                raise ValueError(f"Unexpected seed/protocol: {key}")
            aug = config["augmentation"]
            if aug["name"] != ("baseline" if method == "baseline" else "cutmix"):
                raise ValueError(f"Unexpected method: {key}")
            if method != "baseline" and (aug.get("probability", 1.0) != expected or aug["alpha"] != 1.0):
                raise ValueError(f"Unexpected CutMix settings: {key}")
            settings = normalized(config)
            settings.pop("augmentation")
            if reference is not None and settings != reference:
                raise ValueError(f"Training settings differ: {key}")
            reference = settings
            digest = file_hash(directory / "split_indices.json")
            if split is not None and digest != split:
                raise ValueError("Split mismatch")
            split = digest
            if dataset == "cifar100":
                metadata = json.loads((directory / "robustness/metadata.json").read_text())
                identity = metadata.get("data_sha256", {})
                if len(identity) != 16 or metadata.get("dataset") != dataset or (data_identity is not None and identity != data_identity):
                    raise ValueError("Missing or different CIFAR-100-C data identity")
                data_identity = identity
            history = pd.read_csv(directory / "history.csv")
            if history.epoch.tolist() != list(range(1, config["training"]["epochs"] + 1)):
                raise ValueError(f"Incomplete training: {key}")
            checkpoint = torch.load(Path("checkpoints") / key / "best_model.pt", map_location="cpu", weights_only=True)
            if checkpoint["config"] != config:
                raise ValueError(f"Checkpoint config mismatch: {key}")
            if method == "cutmix_p05":
                if file_hash(directory / "source.zip") != config["run"]["source_sha256"]:
                    raise ValueError("Source archive changed")
                if not ((history.cutmix_batches >= 0) & (history.cutmix_batches <= history.total_batches)).all():
                    raise ValueError("Invalid application counts")
            detail = pd.read_csv(directory / "robustness/corruption_results.csv")
            validate_detail(detail)
            details.append(detail.assign(method=method, seed=seed))
            noise = detail[detail.corruption.isin(["gaussian_noise", "shot_noise"])]
            rows.append(dict(method=method, seed=seed, run=key, clean=checkpoint["test_accuracy"],
                             mca=detail.accuracy.mean(), noise=noise.accuracy.mean(),
                             applied_fraction=(history.cutmix_batches.sum() / history.total_batches.sum()) if method == "cutmix_p05" else (1.0 if method == "cutmix_p1" else 0.0)))
    frame = pd.DataFrame(rows)
    summary = frame.groupby("method")[["clean", "mca", "noise"]].agg(["mean", "std"])
    deltas = []
    p05 = frame[frame.method == "cutmix_p05"].set_index("seed")
    for control in ("baseline", "cutmix_p1"):
        other = frame[frame.method == control].set_index("seed")
        for seed in (42, 43, 44):
            deltas.append(dict(control=control, seed=seed, **{metric: p05.loc[seed, metric] - other.loc[seed, metric] for metric in ("clean", "mca", "noise")}))
    output = Path("results/analysis") / f"{prefix}cutmix-probability"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed.csv", index=False)
    summary.to_csv(output / "mean_std.csv")
    paired = pd.DataFrame(deltas)
    paired.to_csv(output / "paired_deltas.csv", index=False)
    paired.groupby("control")[["clean", "mca", "noise"]].agg(["mean", "std"]).to_csv(output / "paired_mean_std.csv")
    analyze(pd.concat(details, ignore_index=True), output)
    print(summary.to_string())
    print(f"Saved to {output}; SD is sample SD across seeds, not a confidence interval.")


if __name__ == "__main__":
    main()
