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
        
        # Convert mel to waveform (you'd use a vocoder here)
        waveform = self._mel_to_waveform(mel_outputs.squeeze(0))
        
        # Save if path provided
        if output_path:
            self._save_audio(waveform, output_path)
            
        return waveform
    
    def _text_to_tokens(self, text: str) -> list:
        """Convert text to token sequence (simplified)."""
        # This is a placeholder - you'd use a proper tokenizer
        return [ord(c) % 100 for c in text[:50]]  # Simple character-based tokenization
    
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
    
    def _mel_to_waveform(self, mel: Tensor) -> np.ndarray:
        """Convert mel spectrogram to waveform (placeholder)."""
        # This is a placeholder - you'd use a proper vocoder like Griffin-Lim or WaveNet
        # For now, return random audio
        return np.random.randn(16000)  # 1 second of random audio
    
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