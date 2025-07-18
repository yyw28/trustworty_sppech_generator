import os
from os import path
from typing import Optional, List, Dict, Any

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class TrustworthySpeechDataset(Dataset):
    def __init__(
        self, data_list: List[Dict[str, Any]], idxs: list[int], sr: int = 16000
    ):
        self.data_list = data_list
        self.idxs = idxs
        self.sr = sr

        self.resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=sr)
        self.missing_count = 0

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i: int, max_attempts=10):
        import warnings

        attempts = 0
        while attempts < max_attempts:
            data = self.data_list[self.idxs[i]]
            file_path = data["file_path"]
            
            # Get the full path to the audio file
            base_dir = path.dirname(path.dirname(path.dirname(path.dirname(__file__))))
            full_audio_path = path.join(base_dir, file_path)
            
            if not path.exists(full_audio_path):
                self.missing_count += 1
                warnings.warn(f"Audio file not found: {full_audio_path}")
            else:
                try:
                    wav, orig_sr = torchaudio.load(full_audio_path)
                    
                    # Resample if necessary
                    if orig_sr != self.sr:
                        wav = self.resample(wav)

                    # Convert to mono if stereo
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)

                    # Create attention mask (all True for now)
                    mask = torch.ones_like(wav, dtype=torch.bool)
                    
                    # Get trustworthy score from JSON data
                    trustworthy_score = data["trustworthy_score"]
                    # Convert to binary classification if needed
                    # You can adjust this threshold based on your needs
                    trustworthy_binary = torch.tensor(
                        [[trustworthy_score > 0.5]], dtype=torch.float
                    )

                    return wav, mask, trustworthy_binary
                except Exception as e:
                    self.missing_count += 1
                    warnings.warn(f"Could not load audio file: {full_audio_path} ({e})")
            
            i = (i + 1) % len(self)
            attempts += 1
        
        raise RuntimeError("Too many missing or unreadable files in dataset.") 