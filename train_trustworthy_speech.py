#!/usr/bin/env python3
"""
Training script for trustworthy speech classification using HuBERT.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lightning.pytorch.cli import LightningCLI
import torch

from tspeech.model.trustworthiness import TrustworthinessClassifier
from tspeech.data.trustworthy_speech_datamodule import TrustworthySpeechDataModule


def cli_main():
    """Main CLI function for training."""
    torch.set_float32_matmul_precision("high")
    
    # Create CLI with our model and data module
    cli = LightningCLI(
        TrustworthinessClassifier, 
        TrustworthySpeechDataModule,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    cli_main() 