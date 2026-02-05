"""Speaker encoder for voice cloning."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    """Speaker encoder for extracting speaker embeddings from reference audio.

    Takes codec tokens from reference audio and produces a fixed-size
    speaker embedding through pooling.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_codebooks: int = 8,
        pooling: str = "mean",
    ):
        """Initialize speaker encoder.

        Args:
            vocab_size: Codec vocabulary size
            d_model: Embedding dimension
            n_codebooks: Number of codec codebooks
            pooling: Pooling method - "mean", "attention", "last"
        """
        super().__init__()
        self.d_model = d_model
        self.n_codebooks = n_codebooks
        self.pooling = pooling

        # Embedding for each codebook
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model)
            for _ in range(n_codebooks)
        ])

        # Projection after combining codebooks
        self.proj = nn.Linear(d_model, d_model)

        # Attention pooling (if used)
        if pooling == "attention":
            self.attn_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.attn_proj = nn.Linear(d_model, 1)

        # Initialize
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=d_model**-0.5)

    def forward(
        self,
        ref_tokens: torch.Tensor,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract speaker embedding from reference tokens.

        Args:
            ref_tokens: Reference codec tokens [B, K, L] or [B, L] (if K=1)
            ref_mask: Padding mask [B, L], True = valid

        Returns:
            Speaker embedding [B, D]
        """
        # Handle single codebook case
        if ref_tokens.dim() == 2:
            ref_tokens = ref_tokens.unsqueeze(1)

        B, K, L = ref_tokens.shape

        # Embed each codebook and sum
        embeddings = []
        for k in range(min(K, self.n_codebooks)):
            emb = self.embeddings[k](ref_tokens[:, k, :])  # [B, L, D]
            embeddings.append(emb)

        # Sum embeddings across codebooks
        x = sum(embeddings)  # [B, L, D]

        # Apply projection
        x = self.proj(x)

        # Pool to single vector
        if self.pooling == "mean":
            if ref_mask is not None:
                # Masked mean
                mask_expanded = ref_mask.unsqueeze(-1).float()  # [B, L, 1]
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)

        elif self.pooling == "attention":
            # Attention-weighted pooling
            scores = self.attn_proj(x).squeeze(-1)  # [B, L]
            if ref_mask is not None:
                scores = scores.masked_fill(~ref_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)  # [B, L]
            x = (x * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        elif self.pooling == "last":
            if ref_mask is not None:
                # Get last valid position
                lengths = ref_mask.sum(dim=1) - 1  # [B]
                x = x[torch.arange(B, device=x.device), lengths]
            else:
                x = x[:, -1, :]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return x  # [B, D]


class SpeakerEmbeddingTable(nn.Module):
    """Lookup table for known speaker embeddings.

    Used for multi-speaker training with known speaker IDs.
    """

    def __init__(
        self,
        n_speakers: int,
        d_model: int,
    ):
        """Initialize speaker embedding table.

        Args:
            n_speakers: Number of speakers
            d_model: Embedding dimension
        """
        super().__init__()
        self.n_speakers = n_speakers
        self.d_model = d_model

        self.embedding = nn.Embedding(n_speakers, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, speaker_ids: torch.Tensor) -> torch.Tensor:
        """Get speaker embeddings.

        Args:
            speaker_ids: Speaker IDs [B]

        Returns:
            Speaker embeddings [B, D]
        """
        return self.embedding(speaker_ids)


class HybridSpeakerEncoder(nn.Module):
    """Hybrid speaker encoder combining embedding table and reference encoder.

    Uses speaker ID during training (more stable) but can fall back to
    reference-based encoding for zero-shot inference.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_speakers: int,
        n_codebooks: int = 8,
        pooling: str = "mean",
    ):
        """Initialize hybrid speaker encoder.

        Args:
            vocab_size: Codec vocabulary size
            d_model: Embedding dimension
            n_speakers: Number of known speakers
            n_codebooks: Number of codec codebooks
            pooling: Pooling method for reference encoder
        """
        super().__init__()

        self.speaker_table = SpeakerEmbeddingTable(n_speakers, d_model)
        self.ref_encoder = SpeakerEncoder(
            vocab_size, d_model, n_codebooks, pooling
        )

        # Projection to align both representations
        self.align_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        speaker_ids: Optional[torch.Tensor] = None,
        ref_tokens: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get speaker embedding from ID or reference.

        Args:
            speaker_ids: Speaker IDs [B] (for training with known speakers)
            ref_tokens: Reference tokens [B, K, L] (for zero-shot)
            ref_mask: Reference mask [B, L]

        Returns:
            Speaker embedding [B, D]
        """
        if speaker_ids is not None:
            emb = self.speaker_table(speaker_ids)
        elif ref_tokens is not None:
            emb = self.ref_encoder(ref_tokens, ref_mask)
        else:
            raise ValueError("Must provide either speaker_ids or ref_tokens")

        return self.align_proj(emb)
