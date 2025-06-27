from typing import NamedTuple

from torch import Tensor


class TrustworthinessAudioBatch(NamedTuple):
    wav: Tensor # (batch_size, sequence_length) floating-point mono 16 kHz audio waveform data
    mask: Tensor # (batch_size, sequence_length) Boolean mask (True: not masked)
    trustworthy: Tensor # (batch_size, 1) Boolean trustworthy/nontrustworthy
