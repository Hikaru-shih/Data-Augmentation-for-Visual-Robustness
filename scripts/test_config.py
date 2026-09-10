from pprint import pprint

from src.utils.config import load_config


def main():
    config = load_config("configs/baseline.yaml")

    pprint(config)

    print()
    print("Experiment:", config["experiment"]["name"])
    print("Seed:", config["seed"])
    print("Epochs:", config["training"]["epochs"])
    print("Batch size:", config["training"]["batch_size"])
    print("Learning rate:", config["optimizer"]["learning_rate"])


if __name__ == "__main__":
    main()