from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    experiment_name = parser.parse_args().run

    history_path = Path("results") / experiment_name / "history.csv"
    figure_dir = Path("results") / experiment_name / "figures"

    figure_dir.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        raise FileNotFoundError(
            f"History file not found: {history_path}"
        )

    history = pd.read_csv(history_path)
    evaluation = "validation" if "validation_accuracy" in history else "test"

    # --------------------------------------------------
    # Accuracy Curve
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history["epoch"],
        history[f"{evaluation}_accuracy"],
        label=f"{evaluation.title()} Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Training and {evaluation.title()} Accuracy")
    plt.legend()
    plt.grid(True)

    accuracy_path = figure_dir / "accuracy_curve.png"

    plt.savefig(
        accuracy_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # Loss Curve
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history["epoch"],
        history[f"{evaluation}_loss"],
        label=f"{evaluation.title()} Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and {evaluation.title()} Loss")
    plt.legend()
    plt.grid(True)

    loss_path = figure_dir / "loss_curve.png"

    plt.savefig(
        loss_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # Learning Rate Curve
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["learning_rate"],
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)

    lr_path = figure_dir / "learning_rate_curve.png"

    plt.savefig(
        lr_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Plots saved:")
    print(" -", accuracy_path)
    print(" -", loss_path)
    print(" -", lr_path)


if __name__ == "__main__":
    main()
