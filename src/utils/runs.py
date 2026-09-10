import hashlib
import json
import re
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reserve_run(experiment, seed, run_id):
    for value in (experiment, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Experiment and run ID must use letters, numbers, _ or -")
    key = f"{experiment}/seed_{seed}/{run_id}"
    result, checkpoint = Path("results") / key, Path("checkpoints") / key
    if result.exists() or checkpoint.exists():
        raise FileExistsError(f"Run already exists: {key}; choose a new --run-id")
    result.mkdir(parents=True, exist_ok=False)
    checkpoint.mkdir(parents=True, exist_ok=False)
    return key, result, checkpoint


def validate_summary(rows):
    from scripts.evaluate_corruptions import CORRUPTIONS
    names = [row["corruption"] for row in rows]
    if len(names) != 16 or set(names) != set(CORRUPTIONS) | {"overall_mean"}:
        raise ValueError("Expected all 15 corruptions and one overall_mean")
    values = {row["corruption"]: float(row["mean_accuracy"]) for row in rows}
    if not all(0 <= value <= 100 for value in values.values()):
        raise ValueError("Invalid accuracy")
    mean = sum(values[name] for name in CORRUPTIONS) / 15
    if abs(mean - values["overall_mean"]) > 1e-6:
        raise ValueError("Incorrect overall mean")


def verify_evaluation(experiment):
    import csv
    directory = Path("results") / experiment / "robustness"
    with open(directory / "corruption_summary.csv", newline="") as stream:
        validate_summary(list(csv.DictReader(stream)))
    metadata = directory / "metadata.json"
    if not metadata.exists():
        raise ValueError(f"Missing evaluation provenance for {experiment}; rerun evaluation")
    record = json.loads(metadata.read_text(encoding="utf-8"))
    checkpoint = Path("checkpoints") / experiment / "best_model.pt"
    if record["checkpoint_sha256"] != file_hash(checkpoint):
        raise ValueError(f"Stale evaluation for {experiment}")
    for name in ("corruption_results.csv", "corruption_summary.csv"):
        if record["files"][name] != file_hash(directory / name):
            raise ValueError(f"Changed evaluation file: {name}")
