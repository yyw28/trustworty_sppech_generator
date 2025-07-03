from typing import List, Tuple

import torch
from torch import Tensor


def collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Collate function for the trustworthy speech dataset.
    
    Parameters
    ----------
    batch : List[Tuple[Tensor, Tensor, Tensor]]
        List of (wav, mask, trustworthy) tuples
        
    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        Batched (wav, mask, trustworthy) tensors
    """
    wavs, masks, trustworthys = zip(*batch)
    
    # Find maximum length in the batch
    max_length = max(wav.shape[1] for wav in wavs)
    
    # Pad all sequences to the maximum length
    padded_wavs = []
    padded_masks = []
    
    for wav, mask in zip(wavs, masks):
        # Pad wav
        if wav.shape[1] < max_length:
            padding = torch.zeros(1, max_length - wav.shape[1])
            wav = torch.cat([wav, padding], dim=1)
        
        # Pad mask
        if mask.shape[1] < max_length:
            padding = torch.zeros(1, max_length - mask.shape[1], dtype=torch.bool)
            mask = torch.cat([mask, padding], dim=1)
        
        padded_wavs.append(wav)
        padded_masks.append(mask)
    
    # Stack all tensors
    wav_batch = torch.cat(padded_wavs, dim=0)
    mask_batch = torch.cat(padded_masks, dim=0)
    trustworthy_batch = torch.cat(trustworthys, dim=0)
    
    return wav_batch, mask_batch, trustworthy_batch 