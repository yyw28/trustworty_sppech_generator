import os
from os import path
from typing import Optional

import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class SynthSpeechDataset(Dataset):
    def __init__(self, data_dir: str, sr: int = 16000):
        self.data_dir = data_dir
        self.sr = sr

        dfs = []
        for audio_dir, csv in [
            # ("recommendation_humor_audio", "processed_df_rec_hum.csv"),
            # ("recommendation_polite_audio", "processed_df_rec_pol.csv"),
            ("recommendation_neutral_audio", "processed_results_df_rec.csv"),
        ]:
            df = pd.read_csv(path.join(data_dir, "collected_ratings", csv))
            df.filename = (
                path.join(data_dir, audio_dir)
                + "/"
                + "q"
                + (
                    df.filename.str.split("_").str[:2].str.join("_")
                    + "_saved_audio_files_wav/"
                    + df.filename
                )
            )
            df = df[df.filename.str.contains("x-")]
            df = df[~df.filename.str.contains("6_F_intensity_x-soft_pitch_medium")]
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)
        self.resample = torchaudio.transforms.Resample(orig_freq=22050, new_freq=sr)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        data = self.df.iloc[i]
        filename = data.filename

        if not filename.endswith(".wav"):
            filename += ".wav"

        # Load and resample
        wav, sr = torchaudio.load(filename)
        if sr != 22050:
            raise Exception(f"Expected a sample rate of 22050, but found {sr}!")

        wav = self.resample(wav)

        # Create attention mask (all True for now)
        mask = torch.ones_like(wav, dtype=torch.bool)

        # Get trustworthy score
        trustworthy_score = torch.tensor(
            [[(data["trustworthy"] / 5) > 0.5]], dtype=torch.float
        )

        return wav, mask, trustworthy_score
