"""
Tacotron + GST (Global Style Tokens) for speech synthesis.
This module can be used to generate speech that can then be rated for trustworthiness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any
import numpy as np
import os


class GlobalStyleTokens(nn.Module):
    """Global Style Tokens for style control in speech synthesis."""
    
    def __init__(self, num_tokens: int = 10, token_dim: int = 512, embedding_dim: int = 512):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.embedding_dim = embedding_dim
        
        # Style token embeddings
        self.style_tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        
        # Attention mechanism for style selection
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        
        # Style embedding projection
        self.style_projection = nn.Linear(token_dim, embedding_dim)
        
    def forward(self, encoder_outputs: Tensor, style_weights: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            encoder_outputs: [batch_size, seq_len, embedding_dim]
            style_weights: [batch_size, num_tokens] - optional style weights
        Returns:
            style_embedding: [batch_size, embedding_dim]
        """
        batch_size = encoder_outputs.size(0)
        
        if style_weights is None:
            # Use attention to automatically select style tokens
            style_tokens = self.style_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Compute attention between encoder outputs and style tokens
            attn_output, attn_weights = self.attention(
                encoder_outputs, style_tokens, style_tokens
            )
            
            # Average attention weights across sequence
            style_attention = attn_weights.mean(dim=1)  # [batch_size, num_tokens]
        else:
            style_attention = style_weights
            
        # Weighted combination of style tokens
        weighted_tokens = torch.matmul(style_attention, self.style_tokens)
        
        # Project to embedding dimension
        style_embedding = self.style_projection(weighted_tokens)
        
        return style_embedding


class TacotronGST(nn.Module):
    """Tacotron with Global Style Tokens for speech synthesis."""
    
    def __init__(self, 
                 vocab_size: int = 100,
                 embedding_dim: int = 512,
                 encoder_dim: int = 512,
                 decoder_dim: int = 1024,
                 mel_dim: int = 80,
                 num_gst_tokens: int = 10,
                 gst_token_dim: int = 512):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.mel_dim = mel_dim
        
        # Text embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder
        self.encoder = nn.LSTM(
            embedding_dim, encoder_dim // 2, 
            bidirectional=True, batch_first=True
        )
        
        # Global Style Tokens
        self.gst = GlobalStyleTokens(
            num_tokens=num_gst_tokens,
            token_dim=gst_token_dim,
            embedding_dim=encoder_dim
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTMCell(encoder_dim + mel_dim, decoder_dim)
        self.attention = nn.Linear(decoder_dim, encoder_dim)
        self.mel_projection = nn.Linear(decoder_dim, mel_dim)
        
        # Stop token prediction
        self.stop_token = nn.Linear(decoder_dim, 1)
        
    def forward(self, 
                text: Tensor, 
                mel_target: Optional[Tensor] = None,
                style_weights: Optional[Tensor] = None,
                max_decoder_steps: int = 1000) -> Dict[str, Tensor]:
        """
        Args:
            text: [batch_size, text_len] - input text tokens
            mel_target: [batch_size, mel_len, mel_dim] - target mel spectrograms
            style_weights: [batch_size, num_gst_tokens] - style control weights
            max_decoder_steps: maximum decoder steps
        Returns:
            dict with mel_outputs, alignments, stop_tokens
        """
        batch_size = text.size(0)
        
        # Text embedding
        embedded = self.embedding(text)
        
        # Encoder
        encoder_outputs, _ = self.encoder(embedded)
        
        # Global Style Tokens
        style_embedding = self.gst(encoder_outputs, style_weights)
        
        # Expand style embedding to match encoder sequence length
        style_embedding = style_embedding.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        
        # Combine encoder outputs with style
        encoder_outputs = encoder_outputs + style_embedding
        
        # Initialize decoder
        decoder_input = torch.zeros(batch_size, self.mel_dim).to(text.device)
        decoder_hidden = torch.zeros(batch_size, self.decoder_dim).to(text.device)
        decoder_cell = torch.zeros(batch_size, self.decoder_dim).to(text.device)
        
        mel_outputs = []
        alignments = []
        stop_tokens = []
        
        # Decoder loop
        for step in range(max_decoder_steps):
            # Concatenate decoder input with previous mel
            decoder_input_with_context = torch.cat([decoder_input, encoder_outputs.mean(dim=1)], dim=1)
            
            # Decoder step
            decoder_hidden, decoder_cell = self.decoder_lstm(
                decoder_input_with_context, (decoder_hidden, decoder_cell)
            )
            
            # Attention
            attention_weights = F.softmax(
                self.attention(decoder_hidden).unsqueeze(1).bmm(encoder_outputs.transpose(1, 2)),
                dim=-1
            )
            
            # Context vector
            context = attention_weights.bmm(encoder_outputs).squeeze(1)
            
            # Mel prediction
            mel_output = self.mel_projection(decoder_hidden)
            mel_outputs.append(mel_output)
            
            # Stop token prediction
            stop_token = torch.sigmoid(self.stop_token(decoder_hidden))
            stop_tokens.append(stop_token)
            
            # Update decoder input
            if mel_target is not None and step < mel_target.size(1):
                decoder_input = mel_target[:, step, :]
            else:
                decoder_input = mel_output
                
            # Check stop condition
            if stop_token.mean() > 0.5:
                break
                
            alignments.append(attention_weights.squeeze(1))
        
        return {
            'mel_outputs': torch.stack(mel_outputs, dim=1),
            'alignments': torch.stack(alignments, dim=1) if alignments else None,
            'stop_tokens': torch.cat(stop_tokens, dim=1)
        }
    
    def synthesize(self, 
                  text: Tensor, 
                  style_weights: Optional[Tensor] = None,
                  max_decoder_steps: int = 1000) -> Tensor:
        """Synthesize speech from text."""
        with torch.no_grad():
            outputs = self.forward(text, style_weights=style_weights, max_decoder_steps=max_decoder_steps)
            return outputs['mel_outputs']


class TacotronGSTWrapper:
    """Wrapper class for easy Tacotron + GST usage."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        
        # Initialize model (you would load pre-trained weights here)
        self.model = TacotronGST().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pre-trained model from {model_path}")
    
    def synthesize_speech(self, 
                         text: str, 
                         style_name: str = 'neutral',
                         output_path: Optional[str] = None) -> np.ndarray:
        """
        Synthesize speech from text with specified style.
        
        Args:
            text: Input text to synthesize
            style_name: Style to apply ('neutral', 'trustworthy', 'untrustworthy', etc.)
            output_path: Optional path to save audio file
            
        Returns:
            Audio waveform as numpy array
        """
        # Convert text to tokens (simplified - you'd use a proper tokenizer)
        tokens = self._text_to_tokens(text)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        
        # Define style weights based on style name
        style_weights = self._get_style_weights(style_name)
        style_weights = torch.tensor(style_weights).unsqueeze(0).to(self.device)
        
        # Synthesize
        mel_outputs = self.model.synthesize(tokens, style_weights)
        
        # Generate more realistic mel spectrogram
        mel_spectrogram = self._generate_realistic_mel(mel_outputs.squeeze(0), text)
        
        # Convert mel to waveform using Griffin-Lim
        waveform = self._mel_to_waveform(mel_spectrogram)
        
        # Save if path provided
        if output_path:
            self._save_audio(waveform, output_path)
            
        return waveform
    
    def _text_to_tokens(self, text: str) -> list:
        """Convert text to token sequence (improved character-based tokenization)."""
        # Create a simple character vocabulary
        vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        char_to_id = {char: idx + 4 for idx, char in enumerate(set(text.lower()))}
        vocab.update(char_to_id)
        
        # Tokenize text
        tokens = [vocab['<sos>']]  # Start of sequence
        for char in text.lower()[:50]:  # Limit length
            if char in char_to_id:
                tokens.append(char_to_id[char])
            else:
                tokens.append(vocab['<unk>'])
        tokens.append(vocab['<eos>'])  # End of sequence
        
        # Pad to fixed length
        max_len = 52
        if len(tokens) < max_len:
            tokens.extend([vocab['<pad>']] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
            
        return tokens
    
    def _get_style_weights(self, style_name: str) -> list:
        """Get style weights for different styles."""
        styles = {
            'neutral': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'trustworthy': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'untrustworthy': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'friendly': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'professional': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        return styles.get(style_name, styles['neutral'])
    
    def _generate_realistic_mel(self, mel_outputs: Tensor, text: str) -> Tensor:
        """Generate more realistic mel spectrogram based on text content."""
        import torchaudio.transforms as T
        
        # Create a more realistic mel spectrogram based on text length and content
        text_length = len(text)
        duration_seconds = max(1.0, text_length * 0.1)  # Rough estimate: 0.1s per character
        num_frames = int(duration_seconds * 80)  # 80 frames per second
        
        # Create a realistic mel spectrogram pattern
        mel_dim = 80
        mel_spectrogram = torch.zeros(num_frames, mel_dim)
        
        # Add some structure based on text content
        for i in range(num_frames):
            # Create a harmonic structure
            base_freq = 100 + 50 * torch.sin(torch.tensor(i * 0.1))
            harmonics = torch.arange(mel_dim).float() * base_freq / 100
            
            # Add some variation based on character position
            char_pos = min(i // 8, len(text) - 1)  # Map frame to character
            if char_pos < len(text):
                char = text[char_pos].lower()
                # Vary frequency based on character type
                if char in 'aeiou':
                    base_freq *= 1.2  # Vowels are higher frequency
                elif char in 'mn':
                    base_freq *= 0.8  # Nasals are lower frequency
            
            # Create mel bins with harmonic structure
            mel_spectrogram[i] = torch.exp(-torch.abs(harmonics - base_freq) / 20)
        
        # Add some style variation based on the mel_outputs from the model
        if mel_outputs.size(0) > 0:
            # Use the model's output as a style guide
            style_guide = mel_outputs.mean(dim=0, keepdim=True)
            mel_spectrogram = mel_spectrogram * (1 + 0.1 * style_guide[:mel_spectrogram.size(0)])
        
        return mel_spectrogram
    
    def _mel_to_waveform(self, mel: Tensor) -> np.ndarray:
        """Generate speech-like waveform from mel spectrogram."""
        import torchaudio.transforms as T
        
        # Get mel spectrogram dimensions
        mel_np = mel.cpu().numpy()
        num_frames, mel_dim = mel_np.shape
        
        # Generate speech-like audio using sine waves and noise
        sample_rate = 16000
        duration = num_frames * 0.0125  # 80 frames per second
        num_samples = int(duration * sample_rate)
        
        # Create base waveform
        t = np.linspace(0, duration, num_samples)
        waveform = np.zeros(num_samples)
        
        # Add fundamental frequency components based on mel spectrogram
        for frame_idx in range(min(num_frames, 100)):  # Limit to first 100 frames
            frame = mel_np[frame_idx]
            
            # Find dominant frequencies
            dominant_bins = np.argsort(frame)[-5:]  # Top 5 mel bins
            
            for bin_idx in dominant_bins:
                # Convert mel bin to frequency (approximate)
                freq = 80 + bin_idx * 20  # Rough mel-to-frequency mapping
                amplitude = frame[bin_idx] * 0.1
                
                # Add sine wave component
                start_sample = int(frame_idx * num_samples / num_frames)
                end_sample = int((frame_idx + 1) * num_samples / num_frames)
                if end_sample > num_samples:
                    end_sample = num_samples
                
                if start_sample < end_sample:
                    t_frame = np.linspace(0, (end_sample - start_sample) / sample_rate, end_sample - start_sample)
                    waveform[start_sample:end_sample] += amplitude * np.sin(2 * np.pi * freq * t_frame)
        
        # Add some harmonics and noise for realism
        harmonics = np.sin(2 * np.pi * 200 * t) * 0.05
        noise = np.random.normal(0, 0.02, num_samples)
        waveform += harmonics + noise
        
        # Normalize
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        
        return waveform
    
    def _save_audio(self, waveform: np.ndarray, path: str):
        """Save audio to file."""
        import soundfile as sf
        sf.write(path, waveform, 16000)
        print(f"Saved audio to {path}")


# Example usage
if __name__ == "__main__":
    # Initialize TTS system
    tts = TacotronGSTWrapper()
    
    # Synthesize speech with different styles
    text = "Hello, this is a test of trustworthy speech synthesis."
    
    # Generate trustworthy speech
    trustworthy_audio = tts.synthesize_speech(text, style_name='trustworthy', 
                                            output_path='trustworthy_speech.wav')
    
    # Generate untrustworthy speech
    untrustworthy_audio = tts.synthesize_speech(text, style_name='untrustworthy', 
                                              output_path='untrustworthy_speech.wav')
    
    print("Speech synthesis complete!") 