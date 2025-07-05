#!/usr/bin/env python3
"""
SIMPLE DEMO: Basic usage of the Synthesis-to-Trustworthiness Pipeline
"""

import os
import sys
import torch
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tspeech.model.trustworthiness import TrustworthinessClassifier


def simple_demo():
    """Simple demo showing basic usage."""
    print("🎤 SIMPLE SYNTHESIS-TO-TRUSTWORTHINESS DEMO")
    print("=" * 50)
    
    # Check if checkpoint exists
    checkpoint_path = "lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"✅ Using checkpoint: {checkpoint_path}")
    
    # Load the trained model
    try:
        model = TrustworthinessClassifier.load_from_checkpoint(checkpoint_path)
        model.eval()
        print("✅ Model loaded successfully!")
        
        # Force everything to CPU
        device = torch.device("cpu")
        model = model.to(device)
        
        # Test with a dummy audio (you would replace this with real synthesized audio)
        print("\n📊 Testing model with dummy audio...")
        
        # Create dummy audio tensor [batch, time] - 1 second of audio at 16kHz
        dummy_audio = torch.randn(1, 16000).to(device)  # 1 second at 16kHz
        
        # Create attention mask (all True for full sequence)
        mask = torch.ones_like(dummy_audio, dtype=torch.bool).to(device)
        
        with torch.no_grad():
            prediction = model(wav=dummy_audio, mask=mask)
            trustworthiness_score = torch.sigmoid(prediction).item()
        
        print(f"Trustworthiness score: {trustworthiness_score:.4f}")
        print(f"Is trustworthy: {trustworthiness_score > 0.5}")
        
        print("\n🎉 Basic demo completed!")
        print("\nNext steps:")
        print("1. Integrate with a real TTS system (like Tacotron)")
        print("2. Generate audio from text")
        print("3. Rate the generated audio for trustworthiness")
        print("4. Optimize speech synthesis for target trustworthiness scores")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the checkpoint file is valid and all dependencies are installed.")


def test_with_real_audio():
    """Test with a real audio file from your dataset."""
    print("\n" + "=" * 50)
    print("TESTING WITH REAL AUDIO")
    print("=" * 50)
    
    # Find a real audio file from your dataset
    import json
    
    try:
        with open("src/tspeech/data/data_convertion_filter/audio_trustworthy_mapping_filtered.json", 'r') as f:
            data = json.load(f)
        
        if data:
            # Get the first audio file path
            first_item = data[0]
            audio_path = first_item['file_path']
            
            print(f"Testing with audio file: {audio_path}")
            
            if os.path.exists(audio_path):
                # Load the audio
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # Convert to mono if stereo
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Load model
                checkpoint_path = "lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt"
                model = TrustworthinessClassifier.load_from_checkpoint(checkpoint_path)
                model.eval()
                
                # Move to CPU
                device = torch.device("cpu")
                waveform = waveform.to(device)
                mask = torch.ones_like(waveform, dtype=torch.bool).to(device)
                model = model.to(device)
                # Get prediction
                with torch.no_grad():
                    prediction = model(wav=waveform, mask=mask)
                    trustworthiness_score = torch.sigmoid(prediction).item()
                
                print(f"Audio file: {os.path.basename(audio_path)}")
                print(f"Expected trustworthiness: {first_item['trustworthy_score']:.4f}")
                print(f"Predicted trustworthiness: {trustworthiness_score:.4f}")
                print(f"Prediction correct: {abs(trustworthiness_score - first_item['trustworthy_score']) < 0.5}")
                
            else:
                print(f"❌ Audio file not found: {audio_path}")
        else:
            print("❌ No data found in JSON file")
            
    except Exception as e:
        print(f"❌ Error testing with real audio: {e}")


if __name__ == "__main__":
    simple_demo()
    test_with_real_audio() 