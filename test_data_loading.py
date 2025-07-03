#!/usr/bin/env python3
"""
Test script to verify data loading for the trustworthy speech dataset.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from tspeech.data.trustworthy_speech_datamodule import TrustworthySpeechDataModule


def test_data_loading():
    """Test the data loading functionality."""
    
    # Configuration
    csv_file = "collected_ratings/processed_df_rec_hum.csv"
    audio_dir = "Audio/recommendation_humor_audio"
    
    print(f"Testing data loading with:")
    print(f"CSV file: {csv_file}")
    print(f"Audio directory: {audio_dir}")
    print()
    
    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found: {csv_file}")
        return False
    
    if not os.path.exists(audio_dir):
        print(f"ERROR: Audio directory not found: {audio_dir}")
        return False
    
    # Load CSV and show basic info
    df = pd.read_csv(csv_file)
    print(f"CSV loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample filenames: {df['filename'].head().tolist()}")
    print()
    
    # Create data module
    try:
        data_module = TrustworthySpeechDataModule(
            csv_file=csv_file,
            audio_dir=audio_dir,
            batch_size=2,
            num_workers=0  # Use 0 for debugging
        )
        print("Data module created successfully.")
        
        # Setup the data module
        data_module.setup("fit")
        print("Data module setup completed.")
        
        # Test loading a few samples
        train_loader = data_module.train_dataloader()
        print(f"Training dataloader created. Number of batches: {len(train_loader)}")
        
        # Load one batch
        for batch_idx, (wav, mask, trustworthy) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Wav shape: {wav.shape}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Trustworthy shape: {trustworthy.shape}")
            print(f"  Trustworthy values: {trustworthy.flatten().tolist()}")
            print()
            
            if batch_idx >= 2:  # Only test first 3 batches
                break
        
        print("Data loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR during data loading: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n✅ Data loading test passed!")
    else:
        print("\n❌ Data loading test failed!")
        sys.exit(1) 