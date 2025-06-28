import os
from os import path
from typing import Final

import pandas as pd
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tspeech.data.tis_collate_fn import collate_fn
from tspeech.data.tis_dataset import TISDataset


class TISDataModule(LightningDataModule):
    def __init__(self, dataset_dir: str, batch_size: int, num_workers: int):
        super().__init__()

        self.num_workers = num_workers

        # Some of the wav files are missing. Search the ones that are there and use it to narrow down the dataframe
        wav_files: set[str] = set()
        for root, dirs, files in os.walk(path.join(dataset_dir, "Speech WAV Files")):
            for file in files:
                if file.endswith(".wav"):
                    wav_files.add(file.split(".")[0])

        df = pd.read_csv(path.join(dataset_dir, "Speech_dataset_characteristics.csv"))
        df["Audio_Filename"] = df["Audio_Filename"].str.strip()
        df = df[df["Audio_Filename"].isin(wav_files)]
        self.df: Final[pd.DataFrame] = df

        self.dataset_dir: Final[str] = dataset_dir
        self.batch_size: Final[int] = batch_size

        train_ids, test_ids = train_test_split(
            list(self.df.index), train_size=0.8, random_state=42
        )
        val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

    def setup(self, stage: str):
        match stage:
            case "fit":
                self.dataset_train = TISDataset(
                    df=self.df, dataset_dir=self.dataset_dir, idxs=self.train_ids
                )
                self.dataset_validate = TISDataset(
                    df=self.df, dataset_dir=self.dataset_dir, idxs=self.val_ids
                )
            case "validate":
                self.dataset_validate = TISDataset(
                    df=self.df, dataset_dir=self.dataset_dir, idxs=self.val_ids
                )
            case "test":
                self.dataset_test = TISDataset(
                    df=self.df, dataset_dir=self.dataset_dir, idxs=self.test_ids
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
