import os
from os import path
from typing import Final, Literal

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from tspeech.data.synth_speech_dataset import SynthSpeechDataset
from tspeech.data.tis_collate_fn import collate_fn
from tspeech.data.tis_dataset import TISDataset


class TrustworthinessDataModule(LightningDataModule):
    def __init__(
        self,
        datasets: dict[Literal["tis"] | Literal["synth"], str],
        batch_size: int,
        num_workers: int,
    ):

        super().__init__()

        self.num_workers = num_workers
        self.batch_size: Final[int] = batch_size

        if len(datasets) == 0:
            raise Exception("At least one dataset must be specified")

        self.datasets: list[Dataset] = []

        if "tis" in datasets:
            dataset_dir = datasets["tis"]

            # Some of the wav files are missing. Search the ones that are there and use it to narrow down the dataframe
            wav_files: set[str] = set()
            for _, _, files in os.walk(path.join(dataset_dir, "Speech WAV Files")):
                for file in files:
                    if file.endswith(".wav"):
                        wav_files.add(file.split(".")[0])

            df = pd.read_csv(
                path.join(dataset_dir, "Speech_dataset_characteristics.csv")
            )
            df["Audio_Filename"] = df["Audio_Filename"].str.strip()
            df = df[df["Audio_Filename"].isin(wav_files)]

            self.datasets.append(TISDataset(df=df, dataset_dir=dataset_dir))

        if "synth" in datasets:
            dataset_dir = datasets["synth"]
            self.datasets.append(SynthSpeechDataset(data_dir=dataset_dir))

    def setup(self, stage: str):
        self.dataset_train, self.dataset_validate, self.dataset_test = random_split(
            ConcatDataset(self.datasets),
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_validate,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
