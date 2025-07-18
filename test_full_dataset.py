#!/usr/bin/env python3
"""
Test script to verify the full dataset loading and splitting using JSON file.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tspeech.data.trustworthy_speech_datamodule import TrustworthySpeechDataModule


def test_dataset_loading():
    """Test the dataset loading and splitting."""
    print("Testing full dataset loading with JSON file...")
    
    # Create the data module
    datamodule = TrustworthySpeechDataModule(
        batch_size=4,
        num_workers=2,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    print(f"\nDataset summary:")
    print(f"Total samples: {len(datamodule.data_list)}")
    print(f"Train samples: {len(datamodule.train_ids)}")
    print(f"Validation samples: {len(datamodule.val_ids)}")
    print(f"Test samples: {len(datamodule.test_ids)}")
    
    # Test data loading
    print("\nTesting data loading...")
    datamodule.setup("fit")
    
    # Test a few batches
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test one batch from each
    try:
        train_batch = next(iter(train_loader))
        print(f"Train batch shapes: {[t.shape for t in train_batch]}")
        
        val_batch = next(iter(val_loader))
        print(f"Validation batch shapes: {[t.shape for t in val_batch]}")
        
        print("\n✅ Dataset loading test passed!")
        return True
    except Exception as e:
        print(f"\n❌ Dataset loading test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("\nReady to start training!")
    else:
        print("\nPlease fix the issues before training.") 