#!/usr/bin/env python3
"""
Tacotron 2 + GST TTS Pipeline: Convert text to speech and rate trustworthiness
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
import subprocess

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tspeech.model.trustworthiness import TrustworthinessClassifier
from tspeech.tts.tacotron_gst import TacotronGSTWrapper


class TacotronTTSPipeline:
    """A Tacotron 2 + GST TTS pipeline that converts text to speech and rates trustworthiness."""
    
    def __init__(self, hubert_checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Load the trained HuBERT model
        print("Loading HuBERT trustworthiness classifier...")
        
        # Create model instance first
        self.model = TrustworthinessClassifier(
            hubert_model_name="facebook/hubert-base-ls960",
            trainable_layers=10
        )
        
        # Load checkpoint weights
        checkpoint = torch.load(hubert_checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Initialize Tacotron 2 + GST TTS
        print("Initializing Tacotron 2 + GST TTS...")
        self.tts = TacotronGSTWrapper(device=device)
        print("✅ Tacotron 2 + GST TTS initialized!")
        
    def synthesize_with_tacotron(self, text: str, output_path: str, style: str = 'neutral'):
        """Synthesize speech using Tacotron 2 + GST."""
        print(f"Generating speech with Tacotron 2 + GST for style: {style}")
        
        # Map style names to Tacotron styles
        style_mapping = {
            'neutral': 'neutral',
            'trustworthy': 'trustworthy', 
            'untrustworthy': 'untrustworthy',
            'friendly': 'friendly',
            'professional': 'professional'
        }
        
        tacotron_style = style_mapping.get(style, 'neutral')
        
        # Synthesize speech
        waveform = self.tts.synthesize_speech(text, style_name=tacotron_style, output_path=output_path)
        
        print(f"✅ Generated speech with Tacotron 2 + GST: {output_path}")
        return output_path
    
    def synthesize_text(self, text: str, style: str = 'neutral') -> str:
        """Synthesize text to speech using Tacotron 2 + GST."""
        output_path = f"tacotron_{style}.wav"
        return self.synthesize_with_tacotron(text, output_path, style)
    
    def load_and_preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and preprocess for HuBERT."""
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Move to device
        waveform = waveform.to(self.device)
        
        return waveform
    
    def rate_trustworthiness(self, waveform: torch.Tensor) -> float:
        """Rate the trustworthiness of the audio using HuBERT."""
        # Create attention mask
        mask = torch.ones_like(waveform, dtype=torch.bool)
        
        with torch.no_grad():
            prediction = self.model(wav=waveform, mask=mask)
            trustworthiness_score = torch.sigmoid(prediction).item()
        
        return trustworthiness_score
    
    def synthesize_and_rate(self, text: str, style: str = 'neutral') -> dict:
        """Synthesize text to speech and rate its trustworthiness."""
        
        # Synthesize speech
        audio_path = self.synthesize_text(text, style)
        
        # Load and preprocess audio
        waveform = self.load_and_preprocess_audio(audio_path)
        
        # Rate trustworthiness
        trustworthiness_score = self.rate_trustworthiness(waveform)
        
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
            styles = ['neutral', 'trustworthy', 'untrustworthy', 'friendly', 'professional']
        
        print(f"\nComparing styles for: '{text}'")
        print("-" * 50)
        
        results = []
        for style in styles:
            try:
                result = self.synthesize_and_rate(text, style)
                results.append(result)
                
                print(f"{style:12}: {result['trustworthiness_score']:.4f} "
                      f"({'✓' if result['trustworthiness_binary'] else '✗'})")
            except Exception as e:
                print(f"{style:12}: Error - {e}")
        
        if results:
            # Find best and worst styles
            best_style = max(results, key=lambda x: x['trustworthiness_score'])
            worst_style = min(results, key=lambda x: x['trustworthiness_score'])
            
            print("-" * 50)
            print(f"Best style:  {best_style['style']} ({best_style['trustworthiness_score']:.4f})")
            print(f"Worst style: {worst_style['style']} ({worst_style['trustworthiness_score']:.4f})")
        
        return results


def main():
    """Run the Tacotron 2 + GST TTS pipeline demo."""
    print("🎤 TACOTRON 2 + GST TTS-TO-TRUSTWORTHINESS PIPELINE")
    print("=" * 60)
    
    # Check if checkpoint exists
    checkpoint_path = "lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Initialize pipeline
    pipeline = TacotronTTSPipeline(checkpoint_path, device='cpu')
    
    # Demo 1: Basic synthesis and rating
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Text-to-Speech and Rating")
    print("=" * 60)
    
    text = "Hello, I'm here to help you with your request today."
    
    try:
        result = pipeline.synthesize_and_rate(text, style='trustworthy')
        print(f"✅ Audio saved to: {result['audio_path']}")
        print(f"Trustworthiness score: {result['trustworthiness_score']:.4f}")
        print(f"Is trustworthy: {result['trustworthiness_binary']}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Demo 2: Style comparison
    print("\n" + "=" * 60)
    print("DEMO 2: Style Comparison")
    print("=" * 60)
    
    text = "This product is guaranteed to work perfectly for your needs."
    pipeline.compare_styles(text)
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("Generated audio files:")
    for file in os.listdir('.'):
        if file.startswith('tacotron_') and file.endswith('.wav'):
            size = os.path.getsize(file)
            print(f"- {file} ({size} bytes)")


if __name__ == "__main__":
    main() 