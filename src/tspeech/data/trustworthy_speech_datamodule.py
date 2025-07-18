import os
from os import path
from typing import Final, List, Dict, Any
import json

import pandas as pd
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tspeech.data.trustworthy_speech_collate_fn import collate_fn
from tspeech.data.trustworthy_speech_dataset import TrustworthySpeechDataset


class TrustworthySpeechDataModule(LightningDataModule):
    def __init__(self, json_file: str = None, batch_size: int = 4, num_workers: int = 5, 
                 train_ratio: float = 0.5, val_ratio: float = 0.2, test_ratio: float = 0.3,
                 random_state: int = 42):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        # Use the JSON file if provided, otherwise use default
        if json_file is None:
            base_dir = path.dirname(path.dirname(path.dirname(path.dirname(__file__))))
            self.json_file = path.join(base_dir, "src", "tspeech", "data", "data_convertion_filter", 
                                     "audio_trustworthy_mapping_filtered.json")
        else:
            self.json_file = json_file

        # Load and process data from JSON
        self.data_list = self._load_json_data()
        
        # Split the data
        self.train_ids, self.val_ids, self.test_ids = self._split_data()

    def _load_json_data(self) -> List[Dict[str, Any]]:
        """Load data from the JSON file and filter for existing audio files."""
        if not path.exists(self.json_file):
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")
        
        print(f"Loading data from: {self.json_file}")
        
        with open(self.json_file, 'r') as f:
            data_list = json.load(f)
        
        print(f"Loaded {len(data_list)} entries from JSON file")
        
        # Filter for existing audio files
        filtered_data = []
        base_dir = path.dirname(path.dirname(path.dirname(path.dirname(__file__))))
        
        for entry in data_list:
            file_path = entry['file_path']
            full_path = path.join(base_dir, file_path)
            
            if path.exists(full_path):
                filtered_data.append(entry)
            else:
                print(f"Audio file not found: {full_path}")
        
        print(f"Found {len(filtered_data)} existing audio files out of {len(data_list)} entries")
        
        return filtered_data

    def _split_data(self) -> tuple[List[int], List[int], List[int]]:
        """Split data into train, validation, and test sets."""
        total_samples = len(self.data_list)
        
        # Calculate split sizes
        train_size = int(total_samples * self.train_ratio)
        val_size = int(total_samples * self.val_ratio)
        test_size = total_samples - train_size - val_size
        
        print(f"Data split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Split the data
        train_ids, temp_ids = train_test_split(
            list(range(total_samples)), 
            train_size=train_size, 
            random_state=self.random_state
        )
        
        val_ids, test_ids = train_test_split(
            temp_ids, 
            train_size=val_size, 
            random_state=self.random_state
        )
        
        return train_ids, val_ids, test_ids

    def setup(self, stage: str):
        match stage:
            case "fit":
                self.dataset_train = TrustworthySpeechDataset(
                    data_list=self.data_list, idxs=self.train_ids
                )
                self.dataset_validate = TrustworthySpeechDataset(
                    data_list=self.data_list, idxs=self.val_ids
                )
            case "validate":
                self.dataset_validate = TrustworthySpeechDataset(
                    data_list=self.data_list, idxs=self.val_ids
                )
            case "test":
                self.dataset_test = TrustworthySpeechDataset(
                    data_list=self.data_list, idxs=self.test_ids
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