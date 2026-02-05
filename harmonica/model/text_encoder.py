"""Text encoder for converting text to embeddings."""

from typing import Optional

import torch
import torch.nn as nn

from .embedding import TokenEmbedding, SinusoidalPositionalEncoding
from .blocks import TransformerBlock


class TextEncoder(nn.Module):
    """Text encoder using transformer architecture.

    Converts text tokens to contextual embeddings for cross-attention
    in the AR and NAR models.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 512,
        padding_idx: int = 0,
    ):
        """Initialize text encoder.

        Args:
            vocab_size: Text vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            padding_idx: Padding token index
        """
        super().__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx

        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size, d_model, padding_idx=padding_idx
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, max_len, dropout
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text tokens.

        Args:
            text_tokens: Text token indices [B, L]
            text_lengths: Actual lengths [B] (for creating padding mask)

        Returns:
            Tuple of:
                - Text embeddings [B, L, D]
                - Padding mask [B, L], True = masked
        """
        B, L = text_tokens.shape

        # Create padding mask
        if text_lengths is not None:
            padding_mask = torch.arange(L, device=text_tokens.device).expand(B, L) >= text_lengths.unsqueeze(1)
        else:
            padding_mask = text_tokens == self.padding_idx

        # Embed and add positional encoding
        x = self.token_embedding(text_tokens)
        x = self.pos_encoding(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)

        # Final norm
        x = self.norm(x)

        return x, padding_mask


class SimpleTextEncoder(nn.Module):
    """Simplified text encoder without transformer layers.

    Just embedding + positional encoding. Faster but less powerful.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 512,
        padding_idx: int = 0,
    ):
        """Initialize simple text encoder.

        Args:
            vocab_size: Text vocabulary size
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            padding_idx: Padding token index
        """
        super().__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.token_embedding = TokenEmbedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, max_len, dropout
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text tokens.

        Args:
            text_tokens: Text token indices [B, L]
            text_lengths: Actual lengths [B]

        Returns:
            Tuple of (embeddings [B, L, D], padding mask [B, L])
        """
        B, L = text_tokens.shape

        # Create padding mask
        if text_lengths is not None:
            padding_mask = torch.arange(L, device=text_tokens.device).expand(B, L) >= text_lengths.unsqueeze(1)
        else:
            padding_mask = text_tokens == self.padding_idx

        # Embed and add positional encoding
        x = self.token_embedding(text_tokens)
        x = self.pos_encoding(x)
        x = self.norm(x)

        return x, padding_mask
