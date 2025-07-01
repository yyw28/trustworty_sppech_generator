import os
import subprocess
import pandas as pd

def convert_mp3_wav_in_dir(dataset_dir, df):
    """
    Convert all .wav files that are actually MP3s (by content) to true WAV files in-place.
    The new file will have _fixed.wav appended to the name.
    """
    for i, row in df.iterrows():
        filename = row["filename"].strip()
        parts = filename.split("_")
        number = parts[0]
        gender = parts[1]
        subdir = f"q{number}_{gender}_saved_audio_files_wav"
        audio_path = os.path.join(dataset_dir, subdir, f"{filename}.wav")
        if not os.path.exists(audio_path):
            print(f"Missing: {audio_path}")
            continue
        # Check if file is actually MP3
        file_type = subprocess.getoutput(f'file "{audio_path}"')
        if "MPEG ADTS" in file_type or "MP3" in file_type:
            new_path = audio_path.replace('.wav', '_fixed.wav')
            print(f"Converting {audio_path} -> {new_path}")
            cmd = [
                "ffmpeg", "-y", "-i", audio_path, new_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {audio_path}: {e}")
        else:
            print(f"Already WAV: {audio_path}")

if __name__ == "__main__":
    dataset_dir = "/workspaces/trustworty_sppech_generator/Audio/recommendation_netural_audio/"
    csv_path = "/workspaces/trustworty_sppech_generator/collected_ratings/collected_ratings/processed_results_df_rec.csv"
    df = pd.read_csv(csv_path)
    convert_mp3_wav_in_dir(dataset_dir, df)
    print("Conversion complete.")