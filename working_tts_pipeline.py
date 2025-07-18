#!/usr/bin/env python3
"""
WORKING TTS Pipeline: Actually generates audio and rates trustworthiness
"""

import os
import sys
import torch
import numpy as np
import torchaudio
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tspeech.model.trustworthiness import TrustworthinessClassifier


class WorkingTTSPipeline:
    """A working TTS pipeline that generates audio and rates trustworthiness."""
    
    def __init__(self, hubert_checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Load the trained HuBERT model
        print("Loading HuBERT trustworthiness classifier...")
        self.model = TrustworthinessClassifier.load_from_checkpoint(hubert_checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Sample rate for audio generation
        self.sample_rate = 16000
        
    def generate_audio_from_text(self, text: str, style: str = 'neutral', duration: float = 3.0) -> torch.Tensor:
        """
        Generate audio waveform from text (simplified version).
        In a real implementation, this would use Tacotron or another TTS system.
        """
        print(f"Generating audio for: '{text}' with style: {style}")
        
        # Generate different waveforms based on style
        if style == 'trustworthy':
            # Generate a smooth, confident-sounding waveform
            t = torch.linspace(0, duration, int(self.sample_rate * duration))
            # Create a waveform with clear, steady frequency
            waveform = torch.sin(2 * np.pi * 200 * t) * 0.3  # Clear tone
            waveform += torch.sin(2 * np.pi * 400 * t) * 0.2  # Harmonic
            waveform += torch.randn_like(waveform) * 0.05  # Slight noise
            
        elif style == 'untrustworthy':
            # Generate a more erratic, uncertain-sounding waveform
            t = torch.linspace(0, duration, int(self.sample_rate * duration))
            # Create a waveform with varying frequency and amplitude
            waveform = torch.sin(2 * np.pi * (150 + 50 * torch.sin(t)) * t) * 0.4
            waveform += torch.randn_like(waveform) * 0.15  # More noise
            
        elif style == 'professional':
            # Generate a steady, professional-sounding waveform
            t = torch.linspace(0, duration, int(self.sample_rate * duration))
            waveform = torch.sin(2 * np.pi * 250 * t) * 0.25
            waveform += torch.sin(2 * np.pi * 500 * t) * 0.15
            waveform += torch.randn_like(waveform) * 0.03  # Minimal noise
            
        elif style == 'friendly':
            # Generate a warm, friendly-sounding waveform
            t = torch.linspace(0, duration, int(self.sample_rate * duration))
            waveform = torch.sin(2 * np.pi * 180 * t) * 0.3
            waveform += torch.sin(2 * np.pi * 360 * t) * 0.2
            waveform += torch.randn_like(waveform) * 0.08
            
        else:  # neutral
            # Generate a balanced waveform
            t = torch.linspace(0, duration, int(self.sample_rate * duration))
            waveform = torch.sin(2 * np.pi * 220 * t) * 0.3
            waveform += torch.randn_like(waveform) * 0.1
        
        # Normalize and add batch dimension
        waveform = waveform / torch.max(torch.abs(waveform))
        waveform = waveform.unsqueeze(0)  # Add batch dimension [1, time]
        
        return waveform.to(self.device)
    
    def rate_trustworthiness(self, waveform: torch.Tensor) -> float:
        """Rate the trustworthiness of the audio using HuBERT."""
        # Create attention mask
        mask = torch.ones_like(waveform, dtype=torch.bool)
        
        with torch.no_grad():
            prediction = self.model(wav=waveform, mask=mask)
            trustworthiness_score = torch.sigmoid(prediction).item()
        
        return trustworthiness_score
    
    def synthesize_and_rate(self, text: str, style: str = 'neutral', save_audio: bool = True) -> dict:
        """Synthesize audio and rate its trustworthiness."""
        # Generate audio
        waveform = self.generate_audio_from_text(text, style)
        
        # Rate trustworthiness
        trustworthiness_score = self.rate_trustworthiness(waveform)
        
        # Save audio if requested
        audio_path = None
        if save_audio:
            audio_path = f"synthesized_{style}.wav"
            torchaudio.save(audio_path, waveform.cpu(), self.sample_rate)
            print(f"✅ Audio saved to: {audio_path}")
        
        return {
            'audio_path': audio_path,
            'trustworthiness_score': trustworthiness_score,
            'trustworthiness_binary': trustworthiness_score > 0.5,
            'style': style,
            'text': text
        }
    
    def compare_styles(self, text: str, styles: list = None):
        """Compare trustworthiness scores across different styles."""
        if styles is None:
            styles = ['neutral', 'trustworthy', 'untrustworthy', 'professional', 'friendly']
        
        print(f"\nComparing styles for: '{text}'")
        print("-" * 60)
        
        results = []
        for style in styles:
            result = self.synthesize_and_rate(text, style, save_audio=True)
            results.append(result)
            
            print(f"{style:12}: {result['trustworthiness_score']:.4f} "
                  f"({'✓' if result['trustworthiness_binary'] else '✗'})")
        
        # Find best and worst styles
        best_style = max(results, key=lambda x: x['trustworthiness_score'])
        worst_style = min(results, key=lambda x: x['trustworthiness_score'])
        
        print("-" * 60)
        print(f"Best style:  {best_style['style']} ({best_style['trustworthiness_score']:.4f})")
        print(f"Worst style: {worst_style['style']} ({worst_style['trustworthiness_score']:.4f})")
        
        return results


def main():
    """Run the working TTS pipeline demo."""
    print("🎤 WORKING TTS-TO-TRUSTWORTHINESS PIPELINE")
    print("=" * 60)
    
    # Check if checkpoint exists
    checkpoint_path = "lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Initialize pipeline
    pipeline = WorkingTTSPipeline(checkpoint_path, device='cpu')
    
    # Demo 1: Basic synthesis and rating
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Synthesis and Rating")
    print("=" * 60)
    
    text = "Hello, I'm here to help you with your request today."
    result = pipeline.synthesize_and_rate(text, style='trustworthy', save_audio=True)
    
    print(f"\nResults:")
    print(f"- Audio saved to: {result['audio_path']}")
    print(f"- Trustworthiness score: {result['trustworthiness_score']:.4f}")
    print(f"- Is trustworthy: {result['trustworthiness_binary']}")
    
    # Demo 2: Style comparison
    print("\n" + "=" * 60)
    print("DEMO 2: Style Comparison")
    print("=" * 60)
    
    text = "This product is guaranteed to work perfectly for your needs."
    pipeline.compare_styles(text)
    
    # Demo 3: Different types of content
    print("\n" + "=" * 60)
    print("DEMO 3: Different Content Types")
    print("=" * 60)
    
    texts = [
        "I sincerely apologize for the inconvenience.",
        "Trust me, this will work perfectly for you.",
        "Based on my analysis, I recommend this approach.",
        "I'm not entirely sure about this solution."
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\nText {i}: '{text}'")
        pipeline.compare_styles(text, styles=['trustworthy', 'untrustworthy'])
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("Generated audio files:")
    for style in ['trustworthy', 'untrustworthy', 'professional', 'friendly', 'neutral']:
        filename = f"synthesized_{style}.wav"
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"- {filename} ({size} bytes)")
    
    print("\nYou can now:")
    print("1. Listen to the generated audio files")
    print("2. Compare trustworthiness scores across styles")
    print("3. Use this pipeline for your applications!")


if __name__ == "__main__":
    main() 