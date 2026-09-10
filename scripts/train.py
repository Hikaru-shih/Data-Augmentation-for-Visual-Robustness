"""Shared training entry point: validation selection, then one clean test evaluation."""
import argparse
import csv
import json
import platform
import subprocess
from datetime import datetime, timezone

import torch
import torchvision
import yaml

from src.datasets.splits import training_loaders, test_loader
from src.evaluation.evaluator import evaluate
from src.models.resnet import get_resnet18
from src.training.trainer import train_one_epoch
from src.training.mixup_trainer import train_one_epoch_mixup
from src.training.cutmix_trainer import train_one_epoch_cutmix
from src.utils.config import load_config
from src.utils.runs import reserve_run
from src.utils.seed import set_seed
from src.utils.provenance import snapshot_source


def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config, required=default_config is None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    for section, expected in (("dataset", "cifar10"), ("model", "resnet18"),
                              ("optimizer", "sgd"), ("scheduler", "cosine")):
        if config[section]["name"] != expected:
            raise ValueError(f"Unsupported {section}: {config[section]['name']}")
    if config["training"]["epochs"] < 1:
        raise ValueError("epochs must be positive")
    config.setdefault("validation", {"size": 5000, "seed": 2026})
    name = config["augmentation"]["name"]
    if name not in {"baseline", "mixup", "cutmix", "randaugment", "augmix"}:
        raise ValueError(f"Unknown augmentation: {name}")
    set_seed(config["seed"])
    if name == "cutmix":
        probability = config["augmentation"].setdefault("probability", 1.0)
        if not 0 <= probability <= 1:
            raise ValueError("CutMix probability must be in [0, 1]")
    key, result_dir, checkpoint_dir = reserve_run(config["experiment"]["name"], config["seed"], args.run_id)
    config["run"] = {"key": key, "selection": "validation_accuracy", "protocol": "validation-v1"}
    config["run"]["source_sha256"] = snapshot_source(result_dir)
    (result_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, "unavailable"
    environment = dict(python=platform.python_version(), torch=str(torch.__version__),
                       torchvision=str(torchvision.__version__), cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       git_revision=revision, git_status=status)
    (result_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    train, validation, indices = training_loaders(config)
    (result_dir / "split_indices.json").write_text(json.dumps(indices), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18(config["dataset"]["num_classes"]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    opt = config["optimizer"]
    optimizer = torch.optim.SGD(model.parameters(), lr=opt["learning_rate"],
                                momentum=opt["momentum"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    trainer = {"mixup": train_one_epoch_mixup, "cutmix": train_one_epoch_cutmix}.get(name, train_one_epoch)
    kwargs = {"alpha": config["augmentation"]["alpha"]} if name in {"mixup", "cutmix"} else {}
    mixing_stats = {}
    if name == "cutmix":
        kwargs.update(probability=probability, stats=mixing_stats)
    best = -1.0
    checkpoint_path = checkpoint_dir / "best_model.pt"
    with open(result_dir / "history.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "learning_rate", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"] +
                        (["cutmix_batches", "total_batches", "cutmix_fraction"] if name == "cutmix" else []))
        for epoch in range(1, config["training"]["epochs"] + 1):
            loss, accuracy = trainer(model, train, criterion, optimizer, device, **kwargs)
            val_loss, val_accuracy = evaluate(model, validation, criterion, device)
            writer.writerow([epoch, optimizer.param_groups[0]["lr"], loss, accuracy, val_loss, val_accuracy] +
                            ([mixing_stats["applied_batches"], mixing_stats["batches"],
                              mixing_stats["applied_batches"] / mixing_stats["batches"]] if name == "cutmix" else []))
            stream.flush()
            if val_accuracy > best:
                best = val_accuracy
                torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                                validation_accuracy=best, config=config), checkpoint_path)
            scheduler.step()
            print(f"Epoch {epoch}: validation accuracy={val_accuracy:.2f}%", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader(config), criterion, device)
    checkpoint.update(test_loss=test_loss, test_accuracy=test_accuracy)
    torch.save(checkpoint, checkpoint_path)
    (result_dir / "clean_results.json").write_text(json.dumps(dict(
        epoch=checkpoint["epoch"], validation_accuracy=best, test_accuracy=test_accuracy,
        test_loss=test_loss), indent=2), encoding="utf-8")
    print(f"Completed {key}; clean test accuracy={test_accuracy:.2f}%")
    print(f"Evaluate: python -m scripts.evaluate_corruptions {key}")


if __name__ == "__main__":
    main()
