#!/usr/bin/env python3
"""
Simple TTS Pipeline: Convert text to speech and rate trustworthiness
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


class SimpleTTSPipeline:
    """A simple TTS pipeline using gTTS for real speech synthesis."""
    
    def __init__(self, hubert_checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Load the trained HuBERT model
        print("Loading HuBERT trustworthiness classifier...")
        self.model = TrustworthinessClassifier(
            hubert_model_name="facebook/hubert-base-ls960",
            trainable_layers=10
        )
        checkpoint = torch.load(hubert_checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Check TTS availability
        self._check_tts_availability()
    
    def _check_tts_availability(self):
        """Check if gTTS is available."""
        try:
            from gtts import gTTS
            print("✅ gTTS available (real speech)")
        except ImportError:
            print("❌ gTTS not available. Install with: pip install gTTS")
            sys.exit(1)
    
    def synthesize_speech(self, text: str, output_path: str, style: str = 'neutral'):
        """Synthesize speech using gTTS."""
        from gtts import gTTS
        
        print(f"Generating speech for style: {style}")
        
        # Adjust speech rate based on style
        slow_speech = style == 'untrustworthy'
        
        # Create TTS instance
        tts = gTTS(text=text, lang='en', slow=slow_speech)
        
        # Save as MP3 first
        mp3_path = output_path.replace('.wav', '.mp3')
        tts.save(mp3_path)
        
        # Convert MP3 to WAV
        self._convert_mp3_to_wav(mp3_path, output_path)
        
        return output_path
    
    def _convert_mp3_to_wav(self, mp3_path: str, wav_path: str):
        """Convert MP3 to WAV using ffmpeg."""
        try:
            cmd = ['ffmpeg', '-i', mp3_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path, '-y']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Converted to WAV: {wav_path}")
                os.remove(mp3_path)  # Clean up MP3
            else:
                print(f"❌ ffmpeg failed: {result.stderr}")
                
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install ffmpeg: brew install ffmpeg")
        except Exception as e:
            print(f"❌ Conversion error: {e}")
    
    def synthesize_text(self, text: str, style: str = 'neutral') -> str:
        """Synthesize text to speech."""
        output_path = f"speech_{style}.wav"
        return self.synthesize_speech(text, output_path, style)
    
    def load_and_preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and preprocess for HuBERT."""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        return waveform.to(self.device)
    
    def rate_trustworthiness(self, waveform: torch.Tensor) -> float:
        """Rate the trustworthiness of the audio."""
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
            styles = ['neutral', 'trustworthy', 'untrustworthy']
        
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
            best_style = max(results, key=lambda x: x['trustworthiness_score'])
            worst_style = min(results, key=lambda x: x['trustworthiness_score'])
            
            print("-" * 50)
            print(f"Best style:  {best_style['style']} ({best_style['trustworthiness_score']:.4f})")
            print(f"Worst style: {worst_style['style']} ({worst_style['trustworthiness_score']:.4f})")
        
        return results


def main():
    """Run the simple TTS pipeline demo."""
    print("🎤 SIMPLE TTS-TO-TRUSTWORTHINESS PIPELINE")
    print("=" * 50)
    
    # Check if checkpoint exists
    checkpoint_path = "lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Initialize pipeline
    pipeline = SimpleTTSPipeline(checkpoint_path, device='cpu')
    
    # Demo 1: Basic synthesis and rating
    print("\n" + "=" * 50)
    print("DEMO 1: Basic Text-to-Speech and Rating")
    print("=" * 50)
    
    text = "Hello, I'm here to help you with your request today."
    
    try:
        result = pipeline.synthesize_and_rate(text, style='trustworthy')
        print(f"✅ Audio saved to: {result['audio_path']}")
        print(f"Trustworthiness score: {result['trustworthiness_score']:.4f}")
        print(f"Is trustworthy: {result['trustworthiness_binary']}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Demo 2: Style comparison
    print("\n" + "=" * 50)
    print("DEMO 2: Style Comparison")
    print("=" * 50)
    
    text = "This product is guaranteed to work perfectly for your needs."
    pipeline.compare_styles(text)
    
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETE!")
    print("=" * 50)
    print("Generated audio files:")
    for file in os.listdir('.'):
        if file.startswith('speech_') and file.endswith('.wav'):
            size = os.path.getsize(file)
            print(f"- {file} ({size} bytes)")


if __name__ == "__main__":
    main() 