#!/usr/bin/env python3
"""
REAL TTS Pipeline: Uses actual TTS systems to convert text to speech
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
import subprocess
import tempfile

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tspeech.model.trustworthiness import TrustworthinessClassifier


class RealTTSPipeline:
    """A real TTS pipeline that converts text to speech and rates trustworthiness."""
    
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
        
        # Try to import TTS systems
        self.tts_available = self._check_tts_availability()
        
    def _check_tts_availability(self):
        """Check which TTS systems are available."""
        available = {}
        
        # Check pyttsx3 (offline TTS)
        try:
            import pyttsx3
            available['pyttsx3'] = True
            print("✅ pyttsx3 available (offline TTS)")
        except ImportError:
            available['pyttsx3'] = False
            print("❌ pyttsx3 not available. Install with: pip install pyttsx3")
        
        # Check gTTS (Google TTS - online)
        try:
            from gtts import gTTS
            available['gtts'] = True
            print("✅ gTTS available (online TTS)")
        except ImportError:
            available['gtts'] = False
            print("❌ gTTS not available. Install with: pip install gTTS")
        
        # Check pydub for audio conversion
        try:
            from pydub import AudioSegment
            available['pydub'] = True
            print("✅ pydub available (audio conversion)")
        except ImportError:
            available['pydub'] = False
            print("❌ pydub not available. Install with: pip install pydub")
        
        # Check TTS (Coqui TTS)
        try:
            from TTS.api import TTS
            available['coqui'] = True
            print("✅ Coqui TTS available")
        except ImportError:
            available['coqui'] = False
            print("❌ Coqui TTS not available. Install with: pip install TTS")
        
        return available
    
    def synthesize_with_pyttsx3(self, text: str, output_path: str, voice_type: str = 'default'):
        """Synthesize speech using pyttsx3 (offline) with macOS fixes."""
        import pyttsx3
        
        try:
            # Create a new engine instance for each synthesis to avoid run loop issues
            engine = pyttsx3.init()
            
            # Set voice properties based on style
            if voice_type == 'trustworthy':
                # Slower, clearer speech
                engine.setProperty('rate', 150)  # Slower
                engine.setProperty('volume', 0.9)  # Clear
            elif voice_type == 'untrustworthy':
                # Faster, less clear speech
                engine.setProperty('rate', 200)  # Faster
                engine.setProperty('volume', 0.7)  # Quieter
            elif voice_type == 'professional':
                # Balanced, professional
                engine.setProperty('rate', 170)
                engine.setProperty('volume', 0.85)
            else:  # neutral
                engine.setProperty('rate', 180)
                engine.setProperty('volume', 0.8)
            
            # Get available voices
            voices = engine.getProperty('voices')
            if voices:
                # Try to set a specific voice (first available)
                engine.setProperty('voice', voices[0].id)
            
            # Save to file
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            # Clean up
            engine.stop()
            
            return output_path
            
        except Exception as e:
            print(f"pyttsx3 error: {e}")
            # Fallback: try using system say command on macOS
            if os.name == 'posix':  # macOS/Linux
                return self._synthesize_with_say(text, output_path, voice_type)
            else:
                raise e
    
    def _synthesize_with_say(self, text: str, output_path: str, voice_type: str = 'default'):
        """Fallback synthesis using macOS 'say' command, outputs .aiff then converts to .wav."""
        import shutil
        try:
            # Map voice types to different voices
            voice_map = {
                'trustworthy': 'Alex',
                'untrustworthy': 'Daniel',
                'professional': 'Victoria',
                'neutral': 'Alex'
            }
            voice = voice_map.get(voice_type, 'Alex')
            # Use .aiff as intermediate output
            aiff_path = output_path.replace('.wav', '.aiff')
            cmd = [
                'say',
                '-o', aiff_path,
                '-v', voice,
                text
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(aiff_path):
                print(f"✅ Generated audio using macOS 'say' command with voice: {voice}")
                # Convert .aiff to .wav
                wav_path = output_path
                try:
                    if self.tts_available.get('pydub', False):
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(aiff_path, format='aiff')
                        audio.export(wav_path, format='wav')
                        print(f"✅ Converted AIFF to WAV using pydub: {wav_path}")
                        os.remove(aiff_path)
                        return wav_path
                    else:
                        # Try ffmpeg
                        return self._convert_with_ffmpeg(aiff_path, wav_path)
                except Exception as e:
                    print(f"❌ Error converting AIFF to WAV: {e}")
                    # If conversion fails, return the .aiff file
                    return aiff_path
            else:
                raise RuntimeError(f"say command failed: {result.stderr}")
        except Exception as e:
            print(f"say command error: {e}")
            raise RuntimeError(f"Both pyttsx3 and say command failed: {e}")
    
    def synthesize_with_gtts(self, text: str, output_path: str, voice_type: str = 'default'):
        """Synthesize speech using gTTS (online) with proper MP3 to WAV conversion."""
        from gtts import gTTS
        
        # gTTS doesn't have much style control, but we can use different languages/accents
        if voice_type == 'trustworthy':
            # Use a clear, professional accent
            tts = gTTS(text=text, lang='en', slow=False)
        elif voice_type == 'untrustworthy':
            # Use slower speech
            tts = gTTS(text=text, lang='en', slow=True)
        else:
            tts = gTTS(text=text, lang='en', slow=False)
        
        # Save as MP3 first (gTTS default)
        mp3_path = output_path.replace('.wav', '.mp3')
        tts.save(mp3_path)
        
        # Convert MP3 to WAV
        if self.tts_available.get('pydub', False):
            # Use pydub for conversion
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(output_path, format="wav")
                print(f"✅ Converted MP3 to WAV using pydub: {output_path}")
                
                # Clean up the MP3 file
                os.remove(mp3_path)
                
            except Exception as e:
                print(f"❌ pydub conversion failed: {e}")
                # Try ffmpeg directly
                return self._convert_with_ffmpeg(mp3_path, output_path)
        else:
            # Try ffmpeg directly
            return self._convert_with_ffmpeg(mp3_path, output_path)
        
        return output_path
    
    def _convert_with_ffmpeg(self, mp3_path: str, wav_path: str):
        """Convert MP3 to WAV using ffmpeg directly."""
        try:
            cmd = ['ffmpeg', '-i', mp3_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path, '-y']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Converted MP3 to WAV using ffmpeg: {wav_path}")
                # Clean up the MP3 file
                os.remove(mp3_path)
                return wav_path
            else:
                print(f"❌ ffmpeg conversion failed: {result.stderr}")
                # If all conversion fails, just use the MP3 file
                return mp3_path
                
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install ffmpeg:")
            print("  brew install ffmpeg  # macOS")
            print("  apt-get install ffmpeg  # Ubuntu")
            # If ffmpeg not available, just use the MP3 file
            return mp3_path
        except Exception as e:
            print(f"❌ ffmpeg error: {e}")
            return mp3_path
    
    def synthesize_with_coqui(self, text: str, output_path: str, voice_type: str = 'default'):
        """Synthesize speech using Coqui TTS."""
        from TTS.api import TTS
        
        # Initialize TTS with a model
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        
        # Generate speech
        tts.tts_to_file(text=text, file_path=output_path)
        
        return output_path
    
    def synthesize_text(self, text: str, style: str = 'neutral', tts_method: str = 'auto') -> str:
        """Synthesize text to speech using available TTS systems. Always returns a .wav file."""
        # Determine which TTS method to use
        if tts_method == 'auto':
            # Prefer pyttsx3 for offline, then gTTS for online
            if self.tts_available.get('pyttsx3', False):
                tts_method = 'pyttsx3'
            elif self.tts_available.get('gtts', False):
                tts_method = 'gtts'
            elif self.tts_available.get('coqui', False):
                tts_method = 'coqui'
            else:
                raise RuntimeError("No TTS system available. Install pyttsx3, gTTS, or TTS.")
        # Generate output filename
        output_path = f"real_tts_{style}.wav"
        print(f"Generating speech with {tts_method} for style: {style}")
        # Synthesize using the chosen method
        if tts_method == 'pyttsx3' and self.tts_available.get('pyttsx3', False):
            synth_path = self.synthesize_with_pyttsx3(text, output_path, style)
        elif tts_method == 'gtts' and self.tts_available.get('gtts', False):
            synth_path = self.synthesize_with_gtts(text, output_path, style)
        elif tts_method == 'coqui' and self.tts_available.get('coqui', False):
            synth_path = self.synthesize_with_coqui(text, output_path, style)
        else:
            raise RuntimeError(f"TTS method '{tts_method}' not available")
        # Ensure output is .wav, convert if needed
        final_wav = output_path
        if not synth_path.endswith('.wav') or not os.path.exists(synth_path):
            # Try to convert to .wav
            try:
                if self.tts_available.get('pydub', False):
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(synth_path)
                    audio.export(final_wav, format='wav')
                    print(f"✅ Converted {synth_path} to WAV using pydub: {final_wav}")
                    if synth_path != final_wav and os.path.exists(synth_path):
                        os.remove(synth_path)
                else:
                    self._convert_with_ffmpeg(synth_path, final_wav)
            except Exception as e:
                print(f"❌ Could not convert {synth_path} to WAV: {e}")
                final_wav = synth_path
        return final_wav
    
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
    
    def synthesize_and_rate(self, text: str, style: str = 'neutral', tts_method: str = 'auto') -> dict:
        """Synthesize text to speech and rate its trustworthiness."""
        
        # Synthesize speech
        audio_path = self.synthesize_text(text, style, tts_method)
        
        # Load and preprocess audio
        waveform = self.load_and_preprocess_audio(audio_path)
        
        # Rate trustworthiness
        trustworthiness_score = self.rate_trustworthiness(waveform)
        
        return {
            'audio_path': audio_path,
            'trustworthiness_score': trustworthiness_score,
            'trustworthiness_binary': trustworthiness_score > 0.5,
            'style': style,
            'text': text,
            'tts_method': tts_method
        }
    
    def compare_styles(self, text: str, styles: list = None, tts_method: str = 'auto'):
        """Compare trustworthiness scores across different styles."""
        if styles is None:
            styles = ['neutral', 'trustworthy', 'untrustworthy', 'professional']
        
        print(f"\nComparing styles for: '{text}'")
        print("-" * 60)
        
        results = []
        for style in styles:
            try:
                result = self.synthesize_and_rate(text, style, tts_method)
                results.append(result)
                
                print(f"{style:12}: {result['trustworthiness_score']:.4f} "
                      f"({'✓' if result['trustworthiness_binary'] else '✗'})")
            except Exception as e:
                print(f"{style:12}: Error - {e}")
        
        if results:
            # Find best and worst styles
            best_style = max(results, key=lambda x: x['trustworthiness_score'])
            worst_style = min(results, key=lambda x: x['trustworthiness_score'])
            
            print("-" * 60)
            print(f"Best style:  {best_style['style']} ({best_style['trustworthiness_score']:.4f})")
            print(f"Worst style: {worst_style['style']} ({worst_style['trustworthiness_score']:.4f})")
        
        return results


def main():
    """Run the real TTS pipeline demo."""
    print("🎤 REAL TTS-TO-TRUSTWORTHINESS PIPELINE")
    print("=" * 60)
    
    # Check if checkpoint exists
    checkpoint_path = "lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Initialize pipeline
    pipeline = RealTTSPipeline(checkpoint_path, device='cpu')
    
    if not any(pipeline.tts_available.values()):
        print("\n❌ No TTS system available!")
        print("Please install one of the following:")
        print("  pip install pyttsx3    # Offline TTS")
        print("  pip install gTTS       # Online TTS (requires internet)")
        print("  pip install pydub      # Audio conversion")
        print("  pip install TTS        # Coqui TTS")
        return
    
    # Demo 1: Basic synthesis and rating
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Text-to-Speech and Rating")
    print("=" * 60)
    
    text = "Hello, I'm here to help you with your request today."
    
    # Try different TTS methods
    for tts_method in ['pyttsx3', 'gtts', 'coqui']:
        if pipeline.tts_available.get(tts_method, False):
            print(f"\n--- Using {tts_method.upper()} ---")
            try:
                result = pipeline.synthesize_and_rate(text, style='trustworthy', tts_method=tts_method)
                print(f"✅ Audio saved to: {result['audio_path']}")
                print(f"Trustworthiness score: {result['trustworthiness_score']:.4f}")
                print(f"Is trustworthy: {result['trustworthiness_binary']}")
            except Exception as e:
                print(f"❌ Error with {tts_method}: {e}")
    
    # Demo 2: Style comparison with best available TTS
    print("\n" + "=" * 60)
    print("DEMO 2: Style Comparison")
    print("=" * 60)
    
    text = "This product is guaranteed to work perfectly for your needs."
    
    # Use the first available TTS method
    available_method = next((method for method, available in pipeline.tts_available.items() if available), None)
    if available_method:
        pipeline.compare_styles(text, tts_method=available_method)
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("Generated audio files:")
    for file in os.listdir('.'):
        if file.startswith('real_tts_') and (file.endswith('.wav') or file.endswith('.mp3')):
            size = os.path.getsize(file)
            print(f"- {file} ({size} bytes)")
    
    print("\nYou can now:")
    print("1. Listen to the generated speech files")
    print("2. Compare trustworthiness scores across styles")
    print("3. Use different TTS systems for different needs")


if __name__ == "__main__":
    main() 