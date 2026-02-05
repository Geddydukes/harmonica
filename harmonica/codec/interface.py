"""Abstract interface for audio codecs."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class CodecInterface(ABC):
    """Abstract base class for neural audio codecs.

    This interface supports EnCodec, DAC, and other neural codecs
    that use residual vector quantization (RVQ).
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Expected audio sample rate."""
        pass

    @property
    @abstractmethod
    def n_codebooks(self) -> int:
        """Number of codebooks in RVQ."""
        pass

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """Vocabulary size per codebook."""
        pass

    @property
    @abstractmethod
    def hop_length(self) -> int:
        """Samples per codec frame (downsampling factor)."""
        pass

    @abstractmethod
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to discrete tokens.

        Args:
            audio: Waveform tensor [B, T] or [B, 1, T]
                   Expected sample rate: self.sample_rate

        Returns:
            Discrete tokens [B, n_codebooks, S]
            where S = ceil(T / hop_length)
        """
        pass

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens to audio waveform.

        Args:
            tokens: Discrete tokens [B, n_codebooks, S]

        Returns:
            Waveform tensor [B, 1, T]
        """
        pass

    def encode_to_flat(self, audio: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Encode audio and flatten codebooks to single sequence.

        Useful for simplified models that predict all codebooks jointly.

        Args:
            audio: Waveform tensor [B, T]

        Returns:
            Tuple of (flat_tokens [B, S * n_codebooks], original_seq_len)
        """
        tokens = self.encode(audio)  # [B, n_codebooks, S]
        B, K, S = tokens.shape
        flat = tokens.permute(0, 2, 1).reshape(B, S * K)
        return flat, S

    def decode_from_flat(self, flat_tokens: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode flattened tokens back to audio.

        Args:
            flat_tokens: Flattened tokens [B, S * n_codebooks]
            seq_len: Original sequence length S

        Returns:
            Waveform tensor [B, 1, T]
        """
        B = flat_tokens.shape[0]
        K = self.n_codebooks
        tokens = flat_tokens.reshape(B, seq_len, K).permute(0, 2, 1)
        return self.decode(tokens)

    def get_codebook_tokens(
        self, tokens: torch.Tensor, codebook_idx: int
    ) -> torch.Tensor:
        """Extract tokens from a specific codebook.

        Args:
            tokens: All tokens [B, n_codebooks, S]
            codebook_idx: Which codebook (0 = coarsest)

        Returns:
            Tokens from specified codebook [B, S]
        """
        return tokens[:, codebook_idx, :]

    def frames_to_samples(self, n_frames: int) -> int:
        """Convert number of frames to samples."""
        return n_frames * self.hop_length

    def samples_to_frames(self, n_samples: int) -> int:
        """Convert number of samples to frames (ceiling)."""
        return (n_samples + self.hop_length - 1) // self.hop_length
