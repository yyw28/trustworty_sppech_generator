from lightning.pytorch.cli import LightningCLI
import torch

# simple demo classes for your convenience
from tspeech.model import ExampleModel
from tspeech.data import ExampleDataModule


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(ExampleModel, ExampleDataModule)


if __name__ == "__main__":
    cli_main()
