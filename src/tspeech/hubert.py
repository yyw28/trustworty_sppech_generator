from lightning.pytorch.cli import LightningCLI
import torch

# simple demo classes for your convenience
from tspeech.model import HubertModel
from tspeech.data import ExampleDataModule


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(HubertModel, ExampleDataModule)


if __name__ == "__main__":
    cli_main()
