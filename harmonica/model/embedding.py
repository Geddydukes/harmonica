"""Positional and token embeddings."""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in "Attention Is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Tensor with positional encoding added [B, L, D]
        """
        max_len = self.pe.size(1)
        if x.size(1) > max_len:
            x = x[:, :max_len, :]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding using embedding layer."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(max_len, d_model)

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Tensor with positional encoding added [B, L, D]
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            x = x[:, : self.max_len, :]
            seq_len = self.max_len
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.embedding(positions)  # [L, D]
        x = x + pos_emb.unsqueeze(0)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Token embedding with optional scaling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = 0,
        scale: bool = True,
    ):
        """Initialize token embedding.

        Args:
            vocab_size: Vocabulary size
            d_model: Embedding dimension
            padding_idx: Index for padding token
            scale: Scale embeddings by sqrt(d_model)
        """
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.scale_factor = math.sqrt(d_model) if scale else 1.0

        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )

        # Initialize
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**-0.5)
        if padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[padding_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed tokens.

        Args:
            x: Token indices [B, L]

        Returns:
            Embeddings [B, L, D]
        """
        return self.embedding(x) * self.scale_factor


class CodebookEmbedding(nn.Module):
    """Embedding for multiple codebooks (used in NAR model).

    Each codebook has its own embedding table, and embeddings
    are summed across codebooks.
    """

    def __init__(
        self,
        n_codebooks: int,
        vocab_size: int,
        d_model: int,
    ):
        """Initialize codebook embedding.

        Args:
            n_codebooks: Number of codebooks
            vocab_size: Vocabulary size per codebook
            d_model: Embedding dimension
        """
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Separate embedding for each codebook
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model)
            for _ in range(n_codebooks)
        ])

        # Initialize
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=d_model**-0.5)

    def forward(
        self,
        tokens: torch.Tensor,
        codebook_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed tokens from multiple codebooks.

        Args:
            tokens: Token indices [B, K, L] where K is number of codebooks
            codebook_mask: Optional mask [K] for which codebooks to include

        Returns:
            Summed embeddings [B, L, D]
        """
        B, K, L = tokens.shape
        assert K == self.n_codebooks

        # Embed each codebook and sum
        embeddings = []
        for k in range(K):
            if codebook_mask is not None and not codebook_mask[k]:
                continue
            emb = self.embeddings[k](tokens[:, k, :])  # [B, L, D]
            embeddings.append(emb)

        if not embeddings:
            return torch.zeros(B, L, self.d_model, device=tokens.device)

        return sum(embeddings)
