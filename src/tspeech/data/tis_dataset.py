from os import path

import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class TISDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, dataset_dir: str, idxs: list[int], sr: int = 16000
    ):
        self.df = df
        self.dataset_dir = dataset_dir
        self.idxs = idxs

        self.resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=sr)
        self.missing_count = 0

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i: int, max_attempts=10):
        import warnings

        attempts = 0
        while attempts < max_attempts:
            data = self.df.loc[self.idxs[i]].to_dict()
            filename = data["filename"].strip()
            # Parse number and gender from filename
            parts = filename.split("_")
            number = parts[0]
            gender = parts[1]
            subdir = f"q{number}_{gender}_saved_audio_files_wav"
            audio_path = path.join(
                self.dataset_dir,
                subdir,
                f"{filename}.wav",
            )
            if not path.exists(audio_path):
                self.missing_count += 1
                warnings.warn(f"Audio file not found: {audio_path}")
            else:
                try:
                    wav, _ = torchaudio.load(audio_path)
                    wav = self.resample(wav)

                    mask = torch.ones_like(wav, dtype=torch.bool)
                    trustworthy = torch.tensor(
                        [[data["Speaker_Intent"] == "Trustworthy"]], dtype=torch.float
                    )

                    return wav, mask, trustworthy
                except Exception as e:
                    self.missing_count += 1
                    warnings.warn(f"Could not load audio file: {audio_path} ({e})")
            i = (i + 1) % len(self)
            attempts += 1
        raise RuntimeError("Too many missing or unreadable files in dataset.")
