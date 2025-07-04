import os
from os import path
from typing import Final

import pandas as pd
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tspeech.data.trustworthy_speech_collate_fn import collate_fn
from tspeech.data.trustworthy_speech_dataset import TrustworthySpeechDataset


class TrustworthySpeechDataModule(LightningDataModule):
    def __init__(self, csv_file: str, audio_dir: str, batch_size: int, num_workers: int):
        super().__init__()

        self.num_workers = num_workers
        self.csv_file = csv_file
        self.audio_dir = audio_dir
        self.batch_size = batch_size

        # Load the CSV file
        df = pd.read_csv(csv_file)
        
        # Filter out rows where audio files don't exist
        wav_files: set[str] = set()
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith(".wav"):
                    wav_files.add(file)

        # Filter dataframe to only include files that exist
        df = df[df['filename'].isin(wav_files)]
        self.df: Final[pd.DataFrame] = df

        # Split the data
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
                self.dataset_train = TrustworthySpeechDataset(
                    df=self.df, audio_dir=self.audio_dir, idxs=self.train_ids
                )
                self.dataset_validate = TrustworthySpeechDataset(
                    df=self.df, audio_dir=self.audio_dir, idxs=self.val_ids
                )
            case "validate":
                self.dataset_validate = TrustworthySpeechDataset(
                    df=self.df, audio_dir=self.audio_dir, idxs=self.val_ids
                )
            case "test":
                self.dataset_test = TrustworthySpeechDataset(
                    df=self.df, audio_dir=self.audio_dir, idxs=self.test_ids
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
            persistent_workers=self.num_workers > 0,
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
            persistent_workers=self.num_workers > 0,
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
            persistent_workers=self.num_workers > 0,
        ) 