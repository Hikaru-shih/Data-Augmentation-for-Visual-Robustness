import torch
import torchvision


def main():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        z = x @ y

        print("GPU matrix multiplication successful.")
        print("Result shape:", z.shape)


if __name__ == "__main__":
    main()