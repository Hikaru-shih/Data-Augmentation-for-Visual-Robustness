"""Strict aggregation of the planned 3-seed experiment (sample SD, ddof=1)."""
import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.utils.runs import verify_evaluation, file_hash

METHODS = ["baseline", "mixup", "cutmix", "randaugment", "augmix"]


def run_keys():
    return [(method, seed, f"{method}_resnet18/seed_{seed}/" +
             ("validation-v2" if method == "augmix" and seed == 42 else "validation-v1"))
            for method in METHODS for seed in (42, 43, 44)]


def normalized(config):
    result = copy.deepcopy(config)
    for field in ("seed", "run", "experiment"):
        result.pop(field, None)
    return result


def collect():
    rows, configs, splits = [], {}, set()
    missing = [key for _, _, key in run_keys()
               if not (Path("results") / key / "clean_results.json").exists()
               or not (Path("results") / key / "robustness/metadata.json").exists()]
    if missing:
        raise ValueError("Incomplete runs (no partial statistics produced):\n" + "\n".join(missing))
    for method, seed, key in run_keys():
        directory = Path("results") / key
        verify_evaluation(key)
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        if config["seed"] != seed or config["augmentation"]["name"] != method or config["run"]["protocol"] != "validation-v1":
            raise ValueError(f"Unexpected protocol/method/seed: {key}")
        settings = normalized(config)
        if method in configs and configs[method] != settings:
            raise ValueError(f"Configuration changed across seeds: {key}")
        configs[method] = settings
        splits.add(file_hash(directory / "split_indices.json"))
        checkpoint = torch.load(Path("checkpoints") / key / "best_model.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError(f"Checkpoint/config mismatch: {key}")
        history = pd.read_csv(directory / "history.csv")
        if history.epoch.tolist() != list(range(1, config["training"]["epochs"] + 1)):
            raise ValueError(f"Incomplete history: {key}")
        summary = pd.read_csv(directory / "robustness/corruption_summary.csv")
        mca = summary.loc[summary.corruption == "overall_mean", "mean_accuracy"].item()
        rows.append(dict(method=method, seed=seed, run=key,
                         clean_accuracy=checkpoint["test_accuracy"], mca=mca))
    if len(splits) != 1:
        raise ValueError("Runs use different training/validation splits")
    return pd.DataFrame(rows)


def aggregate(frame):
    return frame.groupby("method", sort=False).agg(
        seeds=("seed", "nunique"), clean_mean=("clean_accuracy", "mean"),
        clean_std=("clean_accuracy", "std"), mca_mean=("mca", "mean"), mca_std=("mca", "std"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    frame = collect()
    summary = aggregate(frame)
    if not args.check_only:
        output = Path("results/comparisons/multiseed-42-43-44")
        output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output / "per_seed.csv", index=False)
        summary.to_csv(output / "mean_std.csv")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
