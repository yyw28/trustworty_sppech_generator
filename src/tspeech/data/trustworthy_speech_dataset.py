import os
from os import path
from typing import Optional

import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class TrustworthySpeechDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, audio_dir: str, idxs: list[int], sr: int = 16000
    ):
        self.df = df
        self.audio_dir = audio_dir
        self.idxs = idxs
        self.sr = sr

        self.resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=sr)
        self.missing_count = 0

    def __len__(self) -> int:
        return len(self.idxs)

    def _find_audio_file(self, filename: str) -> Optional[str]:
        """Find the audio file in the nested directory structure."""
        # Remove .wav extension if present
        if filename.endswith('.wav'):
            filename = filename[:-4]
        
        # Search for the file in all subdirectories
        for root, dirs, files in os.walk(self.audio_dir):
            for file in files:
                if file == f"{filename}.wav":
                    return path.join(root, file)
        return None

    def __getitem__(self, i: int, max_attempts=10):
        import warnings

        attempts = 0
        while attempts < max_attempts:
            data = self.df.loc[self.idxs[i]].to_dict()
            filename = data["filename"].strip()
            
            # Find the audio file
            audio_path = self._find_audio_file(filename)
            
            if audio_path is None or not path.exists(audio_path):
                self.missing_count += 1
                warnings.warn(f"Audio file not found: {filename}")
            else:
                try:
                    wav, orig_sr = torchaudio.load(audio_path)
                    
                    # Resample if necessary
                    if orig_sr != self.sr:
                        wav = self.resample(wav)

                    # Convert to mono if stereo
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)

                    # Create attention mask (all True for now)
                    mask = torch.ones_like(wav, dtype=torch.bool)
                    
                    # Get trustworthy score (normalize to 0-1 range if needed)
                    trustworthy_score = data["trustworthy"]
                    # Convert to binary classification if needed
                    # You can adjust this threshold based on your needs
                    trustworthy_binary = torch.tensor(
                        [[trustworthy_score > 0.5]], dtype=torch.float
                    )

                    return wav, mask, trustworthy_binary
                except Exception as e:
                    self.missing_count += 1
                    warnings.warn(f"Could not load audio file: {audio_path} ({e})")
            
            i = (i + 1) % len(self)
            attempts += 1
        
        raise RuntimeError("Too many missing or unreadable files in dataset.") 