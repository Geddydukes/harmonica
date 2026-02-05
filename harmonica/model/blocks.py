"""Transformer building blocks."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Use bias in projections
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None
        self.store_attn = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Query tensor [B, L_q, D]
            key: Key tensor [B, L_k, D]
            value: Value tensor [B, L_k, D]
            attn_mask: Attention mask [L_q, L_k] or [B, L_q, L_k]
            key_padding_mask: Key padding mask [B, L_k], True = masked
            is_causal: Apply causal mask

        Returns:
            Output tensor [B, L_q, D]
        """
        B, L_q, _ = query.shape
        L_k = key.shape[1]

        # Project to Q, K, V
        q = self.q_proj(query)  # [B, L_q, D]
        k = self.k_proj(key)  # [B, L_k, D]
        v = self.v_proj(value)  # [B, L_k, D]

        # Reshape for multi-head attention
        q = q.view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L_q, D_h]
        k = k.view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L_k, D_h]
        v = v.view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L_k, D_h]

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_k]

        # Apply causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(L_q, L_k, device=query.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, L_k], True = masked
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.store_attn:
            self.last_attn_weights = attn_weights.detach()
        else:
            self.last_attn_weights = None
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, H, L_q, D_h]

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        output = self.out_proj(output)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function (gelu, relu)
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor [B, L, D]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
    ):
        """Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
            pre_norm: Use pre-layer normalization (default) vs post-norm
        """
        super().__init__()
        self.pre_norm = pre_norm

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, L, D]
            attn_mask: Attention mask
            key_padding_mask: Key padding mask
            is_causal: Apply causal mask

        Returns:
            Output tensor [B, L, D]
        """
        if self.pre_norm:
            # Pre-layer normalization
            normed = self.norm1(x)
            attn_out = self.self_attn(
                normed, normed, normed,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self.dropout(attn_out)

            normed = self.norm2(x)
            ff_out = self.ff(normed)
            x = x + self.dropout(ff_out)
        else:
            # Post-layer normalization
            attn_out = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )
            x = self.norm1(x + self.dropout(attn_out))

            ff_out = self.ff(x)
            x = self.norm2(x + self.dropout(ff_out))

        return x


class CrossAttentionBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and feed-forward."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
    ):
        """Initialize cross-attention block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
            pre_norm: Use pre-layer normalization
        """
        super().__init__()
        self.pre_norm = pre_norm

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, L, D]
            context: Context tensor for cross-attention [B, L_ctx, D]
            self_attn_mask: Self-attention mask
            self_key_padding_mask: Self-attention key padding mask
            cross_key_padding_mask: Cross-attention key padding mask
            is_causal: Apply causal mask to self-attention

        Returns:
            Output tensor [B, L, D]
        """
        if self.pre_norm:
            # Self-attention
            normed = self.norm1(x)
            attn_out = self.self_attn(
                normed, normed, normed,
                attn_mask=self_attn_mask,
                key_padding_mask=self_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self.dropout(attn_out)

            # Cross-attention
            normed = self.norm2(x)
            cross_out = self.cross_attn(
                normed, context, context,
                key_padding_mask=cross_key_padding_mask,
            )
            x = x + self.dropout(cross_out)

            # Feed-forward
            normed = self.norm3(x)
            ff_out = self.ff(normed)
            x = x + self.dropout(ff_out)
        else:
            # Post-layer normalization (similar structure)
            attn_out = self.self_attn(
                x, x, x,
                attn_mask=self_attn_mask,
                key_padding_mask=self_key_padding_mask,
                is_causal=is_causal,
            )
            x = self.norm1(x + self.dropout(attn_out))

            cross_out = self.cross_attn(
                x, context, context,
                key_padding_mask=cross_key_padding_mask,
            )
            x = self.norm2(x + self.dropout(cross_out))

            ff_out = self.ff(x)
            x = self.norm3(x + self.dropout(ff_out))

        return x
