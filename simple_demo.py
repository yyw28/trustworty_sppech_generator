#!/usr/bin/env python3
"""
Simple Demo: Quick test of TTS and trustworthiness rating
"""

from real_tts_pipeline import SimpleTTSPipeline

def main():
    # Initialize pipeline
    pipeline = SimpleTTSPipeline(
        "lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt"
    )
    
    # Test text
    text = "Hello, this is a test of the speech synthesis system."
    
    # Generate speech and rate it
    result = pipeline.synthesize_and_rate(text, style='neutral')
    
    print(f"Text: {text}")
    print(f"Audio: {result['audio_path']}")
    print(f"Trustworthiness: {result['trustworthiness_score']:.4f}")
    print(f"Trusted: {'Yes' if result['trustworthiness_binary'] else 'No'}")

if __name__ == "__main__":
    main() 