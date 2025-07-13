from lightning.pytorch.cli import LightningCLI
import torch

# simple demo classes for your convenience
from tspeech.model.trustworthiness import TrustworthinessClassifier
from tspeech.data import TrustworthinessDataModule


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(TrustworthinessClassifier, TrustworthinessDataModule)


if __name__ == "__main__":
    cli_main()
