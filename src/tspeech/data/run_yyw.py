import torchaudio
# Set torchaudio backend to sox_io for better wav compatibility
try:
    torchaudio.set_audio_backend("sox_io")
except Exception as e:
    print(f"Warning: Could not set torchaudio backend to sox_io: {e}")

import pandas as pd
from torch.utils.data import DataLoader
from src.tspeech.data.tis_dataset import TISDataset
from src.tspeech.model.trustworthiness import TrustworthinessClassifier
from lightning.pytorch import Trainer
import os

# 1. Load your CSV
df = pd.read_csv("collected_ratings/processed_results_df_rec.csv")  # Update with your CSV path

# Print first 5 expected audio paths and their existence
print("Sample of expected audio paths:")
for i, row in df.head(5).iterrows():
    filename = row["filename"].strip()
    parts = filename.split("_")
    number = parts[0]
    gender = parts[1]
    subdir = f"q{number}_{gender}_saved_audio_files_wav"
    audio_path = os.path.join("/workspaces/trustworty_sppech_generator/Audio/recommendation_polite_audio/", subdir, f"{filename}")
    print(audio_path, "EXISTS" if os.path.exists(audio_path) else "MISSING")

def is_valid_audio(row, dataset_dir):
    filename = row["filename"].strip()
    parts = filename.split("_")
    number = parts[0]
    gender = parts[1]
    subdir = f"q{number}_{gender}_saved_audio_files_wav"
    audio_path = os.path.join(dataset_dir, subdir, f"{filename}.wav")
    if not os.path.exists(audio_path):
        return False
    try:
        torchaudio.load(audio_path)
        return True
    except Exception:
        return False

dataset_dir = "/workspaces/trustworty_sppech_generator/Audio/recommendation_netural_audio"
#dataset_dir = "/workspaces/trustworty_sppech_generator/Audio/recommendation_polite_audio/"
print("Checking audio files, this may take a while...")
valid_mask = df.apply(is_valid_audio, axis=1, dataset_dir=dataset_dir)
print(f"Valid audio files: {valid_mask.sum()} / {len(df)}")
df = df[valid_mask].reset_index(drop=True)

# 2. Split indices for train/val/test
from sklearn.model_selection import train_test_split
idxs = list(range(len(df)))
train_idxs, temp_idxs = train_test_split(idxs, test_size=0.2, random_state=42)
val_idxs, test_idxs = train_test_split(temp_idxs, test_size=0.5, random_state=42)

# 3. Create Datasets
train_dataset = TISDataset(df, dataset_dir, train_idxs)
val_dataset = TISDataset(df, dataset_dir, val_idxs)
test_dataset = TISDataset(df, dataset_dir, test_idxs)

# 4. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# 5. Instantiate the model
model = TrustworthinessClassifier(
    hubert_model_name="facebook/hubert-base-ls960",  # or your preferred checkpoint
    trainable_layers=2
)

# 6. Train
trainer = Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model, train_loader, val_loader)

# 7. Test
trainer.test(model, test_loader)