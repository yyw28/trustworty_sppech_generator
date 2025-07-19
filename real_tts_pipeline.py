#!/usr/bin/env python3
"""
Hybrid TTS Pipeline: Real TTS + Tacotron 2 + GST for style control
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


class HybridTTSPipeline:
    """A hybrid TTS pipeline that combines real TTS with Tacotron 2 + GST style control."""
    
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
        self.tacotron_tts = TacotronGSTWrapper(device=device)
        print("✅ Tacotron 2 + GST TTS initialized!")
        
        # Check for real TTS systems
        self.tts_available = self._check_tts_availability()
        
    def _check_tts_availability(self):
        """Check which real TTS systems are available."""
        available = {}
        
        # Check gTTS (Google TTS - online)
        try:
            from gtts import gTTS
            available['gtts'] = True
            print("✅ gTTS available (real speech)")
        except ImportError:
            available['gtts'] = False
            print("❌ gTTS not available. Install with: pip install gTTS")
        
        # Check pyttsx3 (offline TTS)
        try:
            import pyttsx3
            available['pyttsx3'] = True
            print("✅ pyttsx3 available (offline TTS)")
        except ImportError:
            available['pyttsx3'] = False
            print("❌ pyttsx3 not available. Install with: pip install pyttsx3")
        
        # Check pydub for audio conversion
        try:
            from pydub import AudioSegment
            available['pydub'] = True
            print("✅ pydub available (audio conversion)")
        except ImportError:
            available['pydub'] = False
            print("❌ pydub not available. Install with: pip install pydub")
        
        return available
    
    def synthesize_with_real_tts(self, text: str, output_path: str, style: str = 'neutral'):
        """Synthesize speech using real TTS systems."""
        print(f"Generating REAL speech for style: {style}")
        
        # Try gTTS first (online, better quality)
        if self.tts_available.get('gtts', False):
            try:
                return self._synthesize_with_gtts(text, output_path, style)
            except Exception as e:
                print(f"gTTS failed: {e}")
        
        # Fallback to pyttsx3 (offline)
        if self.tts_available.get('pyttsx3', False):
            try:
                return self._synthesize_with_pyttsx3(text, output_path, style)
            except Exception as e:
                print(f"pyttsx3 failed: {e}")
        
        # Fallback to Tacotron 2 + GST (synthetic)
        print("⚠️ Using Tacotron 2 + GST (synthetic speech)")
        return self.tacotron_tts.synthesize_speech(text, style_name=style, output_path=output_path)
    
    def _synthesize_with_gtts(self, text: str, output_path: str, style: str = 'neutral'):
        """Synthesize speech using gTTS (Google TTS)."""
        from gtts import gTTS
        
        # Adjust speech rate based on style
        slow_speech = style == 'untrustworthy'  # Slower for untrustworthy
        
        # Create TTS instance
        tts = gTTS(text=text, lang='en', slow=slow_speech)
        
        # Save as MP3 first (gTTS default)
        mp3_path = output_path.replace('.wav', '.mp3')
        tts.save(mp3_path)
        
        # Convert MP3 to WAV
        if self.tts_available.get('pydub', False):
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(output_path, format="wav")
                print(f"✅ Converted MP3 to WAV: {output_path}")
                os.remove(mp3_path)  # Clean up MP3
            except Exception as e:
                print(f"❌ pydub conversion failed: {e}")
                self._convert_with_ffmpeg(mp3_path, output_path)
        else:
            self._convert_with_ffmpeg(mp3_path, output_path)
        
        return output_path
    
    def _synthesize_with_pyttsx3(self, text: str, output_path: str, style: str = 'neutral'):
        """Synthesize speech using pyttsx3 (offline TTS)."""
        import pyttsx3
        
        # Initialize TTS engine
        engine = pyttsx3.init()
        
        # Adjust speech rate based on style
        if style == 'untrustworthy':
            engine.setProperty('rate', 150)  # Slower
        elif style == 'trustworthy':
            engine.setProperty('rate', 200)  # Faster
        else:
            engine.setProperty('rate', 175)  # Normal
        
        # Save to file
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        print(f"✅ Generated speech with pyttsx3: {output_path}")
        return output_path
    
    def _convert_with_ffmpeg(self, mp3_path: str, wav_path: str):
        """Convert MP3 to WAV using ffmpeg."""
        try:
            cmd = ['ffmpeg', '-i', mp3_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path, '-y']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Converted MP3 to WAV using ffmpeg: {wav_path}")
                os.remove(mp3_path)  # Clean up MP3
            else:
                print(f"❌ ffmpeg conversion failed: {result.stderr}")
                
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install ffmpeg: brew install ffmpeg")
        except Exception as e:
            print(f"❌ ffmpeg error: {e}")
    
    def synthesize_text(self, text: str, style: str = 'neutral') -> str:
        """Synthesize text to speech using the best available TTS system."""
        output_path = f"real_tts_{style}.wav"
        return self.synthesize_with_real_tts(text, output_path, style)
    
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
    """Run the hybrid TTS pipeline demo."""
    print("🎤 HYBRID TTS-TO-TRUSTWORTHINESS PIPELINE")
    print("=" * 60)
    
    # Check if checkpoint exists
    checkpoint_path = "lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Initialize pipeline
    pipeline = HybridTTSPipeline(checkpoint_path, device='cpu')
    
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
        if file.startswith('real_tts_') and file.endswith('.wav'):
            size = os.path.getsize(file)
            print(f"- {file} ({size} bytes)")


if __name__ == "__main__":
    main() 