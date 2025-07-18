#!/usr/bin/env python3
"""
DEMO: How to use the Synthesis-to-Trustworthiness Pipeline
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synthesis_to_trustworthiness import SynthesisToTrustworthinessPipeline


def demo_basic_usage():
    """Demo 1: Basic usage - synthesize speech and rate trustworthiness."""
    print("=" * 60)
    print("DEMO 1: Basic Usage")
    print("=" * 60)
    
    # Initialize pipeline with your trained HuBERT model
    pipeline = SynthesisToTrustworthinessPipeline(
        hubert_checkpoint_path="lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt",
        device='auto'
    )
    
    # Test text
    text = "Hello, I'm here to help you with your request today."
    
    # Synthesize with trustworthy style
    print(f"Text: '{text}'")
    result = pipeline.synthesize_and_rate(
        text=text,
        style_name='trustworthy',
        save_audio=True
    )
    
    print(f"\nResults:")
    print(f"- Audio saved to: {result['audio_path']}")
    print(f"- Trustworthiness score: {result['trustworthiness_score']:.4f}")
    print(f"- Is trustworthy: {result['trustworthiness_binary']}")


def demo_style_comparison():
    """Demo 2: Compare different speech styles."""
    print("\n" + "=" * 60)
    print("DEMO 2: Style Comparison")
    print("=" * 60)
    
    pipeline = SynthesisToTrustworthinessPipeline(
        hubert_checkpoint_path="lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt",
        device='auto'
    )
    
    text = "This product is guaranteed to work perfectly for your needs."
    
    # Compare different styles
    pipeline.compare_styles(
        text=text,
        styles=['neutral', 'trustworthy', 'untrustworthy', 'professional', 'friendly']
    )


def demo_target_trustworthiness():
    """Demo 3: Generate speech with target trustworthiness score."""
    print("\n" + "=" * 60)
    print("DEMO 3: Target Trustworthiness Generation")
    print("=" * 60)
    
    pipeline = SynthesisToTrustworthinessPipeline(
        hubert_checkpoint_path="lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt",
        device='auto'
    )
    
    text = "I understand your concerns and will address them immediately."
    
    # Try different target scores
    target_scores = [0.3, 0.5, 0.7, 0.9]
    
    for target_score in target_scores:
        print(f"\nTarget trustworthiness: {target_score}")
        result = pipeline.generate_trustworthy_speech(text, target_score=target_score)
        if result:
            print(f"Best style: {result['style']} (score: {result['trustworthiness_score']:.4f})")


def demo_batch_processing():
    """Demo 4: Process multiple texts at once."""
    print("\n" + "=" * 60)
    print("DEMO 4: Batch Processing")
    print("=" * 60)
    
    pipeline = SynthesisToTrustworthinessPipeline(
        hubert_checkpoint_path="lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt",
        device='auto'
    )
    
    # Different types of texts
    texts = [
        "Hello, how can I assist you today?",
        "This is absolutely the best solution available.",
        "I apologize for any inconvenience this may have caused.",
        "Trust me, this will work perfectly for you.",
        "I'm not entirely sure about this approach."
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\nText {i}: '{text}'")
        pipeline.compare_styles(text, styles=['trustworthy', 'untrustworthy'])


def demo_custom_usage():
    """Demo 5: Custom usage examples."""
    print("\n" + "=" * 60)
    print("DEMO 5: Custom Usage Examples")
    print("=" * 60)
    
    pipeline = SynthesisToTrustworthinessPipeline(
        hubert_checkpoint_path="lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt",
        device='auto'
    )
    
    # Example 1: Customer service response
    print("\n1. Customer Service Response:")
    customer_text = "I sincerely apologize for the inconvenience. Let me resolve this issue for you right away."
    result = pipeline.synthesize_and_rate(customer_text, style_name='professional', save_audio=True)
    print(f"Trustworthiness: {result['trustworthiness_score']:.4f}")
    
    # Example 2: Product recommendation
    print("\n2. Product Recommendation:")
    product_text = "This product has been tested extensively and comes with a full warranty."
    result = pipeline.synthesize_and_rate(product_text, style_name='trustworthy', save_audio=True)
    print(f"Trustworthiness: {result['trustworthiness_score']:.4f}")
    
    # Example 3: Medical advice
    print("\n3. Medical Advice:")
    medical_text = "Based on your symptoms, I recommend consulting with a healthcare professional."
    result = pipeline.synthesize_and_rate(medical_text, style_name='professional', save_audio=True)
    print(f"Trustworthiness: {result['trustworthiness_score']:.4f}")


def main():
    """Run all demos."""
    print("🎤 SYNTHESIS-TO-TRUSTWORTHINESS PIPELINE DEMO")
    print("This demo shows how to use your trained HuBERT model to rate synthesized speech.")
    
    try:
        # Check if checkpoint exists
        checkpoint_path = "lightning_logs/trustworthy_speech_full_dataset/version_2/checkpoints/last.ckpt"
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("Please make sure your HuBERT training has completed.")
            return
        
        print(f"✅ Using checkpoint: {checkpoint_path}")
        
        # Run demos
        demo_basic_usage()
        demo_style_comparison()
        demo_target_trustworthiness()
        demo_batch_processing()
        demo_custom_usage()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETE!")
        print("=" * 60)
        print("Generated audio files:")
        print("- synthesized_trustworthy.wav")
        print("- synthesized_professional.wav")
        print("- synthesized_untrustworthy.wav")
        print("\nYou can now:")
        print("1. Listen to the generated audio files")
        print("2. Compare trustworthiness scores across styles")
        print("3. Generate speech with specific trustworthiness targets")
        print("4. Use this pipeline for your applications!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure all dependencies are installed and the model checkpoint exists.")


if __name__ == "__main__":
    main() 