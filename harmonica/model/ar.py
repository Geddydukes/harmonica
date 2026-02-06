"""Autoregressive Transformer for first codebook prediction."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import TokenEmbedding, SinusoidalPositionalEncoding
from .blocks import CrossAttentionBlock
from .text_encoder import TextEncoder
from .speaker import SpeakerEncoder
from .duration_predictor import DurationPredictor


class ARTransformer(nn.Module):
    """Autoregressive Transformer for VALL-E style TTS.

    Predicts codebook 1 tokens autoregressively, conditioned on:
    - Text embeddings (via cross-attention)
    - Speaker prompt (prefix of reference audio tokens)

    Architecture:
    - Token embedding for audio codes
    - Positional encoding
    - Stack of transformer blocks with cross-attention to text
    - Output projection to codebook vocabulary
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        text_vocab_size: int = 128,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        max_text_len: int = 512,
        text_padding_idx: int = 0,
        n_text_layers: int = 4,
        length_control_mode: str = "duration_predictor",
        duration_hidden_dim: int = 256,
        length_prompt_max: int = 500,
    ):
        """Initialize AR Transformer.

        Args:
            vocab_size: Audio codebook vocabulary size
            text_vocab_size: Text vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_seq_len: Maximum audio sequence length
            max_text_len: Maximum text sequence length
            text_padding_idx: Text padding token index
            n_text_layers: Number of text encoder layers
        """
        super().__init__()
        self.length_control_mode = length_control_mode
        self.length_prompt_max = length_prompt_max
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.text_vocab_size = text_vocab_size

        # Optional stop token expands vocabulary by 1
        self.stop_token_id = None
        self.codec_vocab_size = vocab_size
        self.vocab_size = vocab_size
        if self.length_control_mode == "stop_token":
            self.stop_token_id = vocab_size
            self.vocab_size = vocab_size + 1

        self.d_model = d_model
        self.n_layers = n_layers

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=text_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_text_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_text_len,
            padding_idx=text_padding_idx,
        )

        # Audio token embedding (codebook 1)
        self.audio_embedding = TokenEmbedding(self.vocab_size, d_model)

        # Positional encoding for audio
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, max_seq_len, dropout
        )

        # Transformer layers with cross-attention
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, self.vocab_size)

        # Special tokens
        self.bos_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Initialize output projection
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=d_model**-0.5)

        # Length control modules
        if self.length_control_mode == "duration_predictor":
            self.duration_predictor = DurationPredictor(
                text_dim=d_model,
                hidden_dim=duration_hidden_dim,
                dropout=dropout,
            )
        elif self.length_control_mode == "length_prompt":
            self.length_embedding = nn.Embedding(length_prompt_max, d_model)

    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        prompt_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training with teacher forcing.

        Args:
            audio_tokens: Target audio tokens [B, L] (codebook 1)
            text_tokens: Text tokens [B, T]
            text_lengths: Text lengths [B]
            prompt_tokens: Speaker prompt tokens [B, P] (optional)
            prompt_lengths: Prompt lengths [B] (optional)

        Returns:
            Logits [B, L, vocab_size]
        """
        B, L = audio_tokens.shape

        # Encode text
        text_emb, text_mask = self.text_encoder(text_tokens, text_lengths)

        # Embed audio tokens
        audio_emb = self.audio_embedding(audio_tokens)  # [B, L, D]

        # Prepend BOS token
        bos = self.bos_token.expand(B, -1, -1)  # [B, 1, D]

        # Handle prompt (speaker reference)
        if prompt_tokens is not None:
            prompt_emb = self.audio_embedding(prompt_tokens)  # [B, P, D]
            # Prepend prompt before BOS and target
            x = torch.cat([prompt_emb, bos, audio_emb], dim=1)  # [B, P+1+L, D]
            prompt_len = prompt_tokens.shape[1]
        else:
            x = torch.cat([bos, audio_emb], dim=1)  # [B, 1+L, D]
            prompt_len = 0

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create causal mask - attention only to previous positions
        # But we want to shift by 1 to predict next token
        seq_len = x.shape[1]

        # Apply transformer layers with cross-attention to text
        for layer in self.layers:
            x = layer(
                x,
                context=text_emb,
                cross_key_padding_mask=text_mask,
                is_causal=True,
            )

        # Final norm
        x = self.norm(x)

        # Output projection (skip prompt, predict from BOS onward)
        logits = self.output_proj(x[:, prompt_len:, :])  # [B, 1+L, vocab_size]

        # Return logits aligned with targets (remove BOS position prediction)
        # logits[:, 0] predicts audio_tokens[:, 0]
        # logits[:, 1] predicts audio_tokens[:, 1]
        # etc.
        return logits[:, :-1, :]  # [B, L, vocab_size]

    def encode_text(
        self,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text tokens for reuse in decoding."""
        return self.text_encoder(text_tokens, text_lengths)

    def forward_step(
        self,
        prev_tokens: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits for the next token given previous tokens.

        Args:
            prev_tokens: Previous audio tokens [B, L_prev]
            text_emb: Text embeddings [B, T, D]
            text_mask: Text padding mask [B, T]
            prompt_tokens: Speaker prompt tokens [B, P] (optional)

        Returns:
            Logits for next token [B, vocab_size]
        """
        B = text_emb.shape[0]

        bos = self.bos_token.expand(B, -1, -1)

        if prev_tokens is not None and prev_tokens.numel() > 0:
            audio_emb = self.audio_embedding(prev_tokens)
        else:
            audio_emb = None

        if prompt_tokens is not None:
            prompt_emb = self.audio_embedding(prompt_tokens)
            if audio_emb is not None:
                x = torch.cat([prompt_emb, bos, audio_emb], dim=1)
            else:
                x = torch.cat([prompt_emb, bos], dim=1)
        else:
            if audio_emb is not None:
                x = torch.cat([bos, audio_emb], dim=1)
            else:
                x = bos

        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(
                x,
                context=text_emb,
                cross_key_padding_mask=text_mask,
                is_causal=True,
            )

        x = self.norm(x)
        logits = self.output_proj(x[:, -1, :])
        return logits

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate audio tokens autoregressively.

        Args:
            text_tokens: Text tokens [B, T]
            text_lengths: Text lengths [B]
            prompt_tokens: Speaker prompt tokens [B, P]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            eos_token: Stop generation if this token is produced

        Returns:
            Generated tokens [B, L]
        """
        B = text_tokens.shape[0]
        device = text_tokens.device

        # Encode text once
        text_emb, text_mask = self.text_encoder(text_tokens, text_lengths)

        # Determine length control strategy
        target_length = max_length
        stop_condition = None

        if self.length_control_mode == "duration_predictor":
            target_lengths = self.duration_predictor.predict_total_length(text_emb)
            target_length = int(target_lengths.max().item())
            target_length = min(target_length, max_length)
            stop_condition = lambda current_len: current_len >= target_length
        elif self.length_control_mode == "stop_token":
            stop_condition = None
        elif self.length_control_mode == "length_prompt":
            estimated_length = int(text_tokens.shape[1] * 7.5)
            target_length = min(estimated_length, max_length)
            stop_condition = lambda current_len: current_len >= target_length

            length_idx = max(0, min(self.length_prompt_max - 1, target_length))
            length_idx = torch.tensor([length_idx], device=device)
            length_emb = self.length_embedding(length_idx).unsqueeze(1)  # [1, 1, D]
            length_emb = length_emb.expand(B, -1, -1)
            text_emb = torch.cat([length_emb, text_emb], dim=1)
            if text_mask is not None:
                pad = torch.zeros(B, 1, dtype=text_mask.dtype, device=device)
                text_mask = torch.cat([pad, text_mask], dim=1)

        # Start with BOS
        bos = self.bos_token.expand(B, -1, -1)

        # Handle prompt
        if prompt_tokens is not None:
            prompt_emb = self.audio_embedding(prompt_tokens)
            x = torch.cat([prompt_emb, bos], dim=1)
            prompt_len = prompt_tokens.shape[1]
        else:
            x = bos
            prompt_len = 0

        generated = []

        for _ in range(max_length):
            # Add positional encoding
            x_pos = self.pos_encoding(x)

            # Apply transformer layers
            h = x_pos
            for layer in self.layers:
                h = layer(
                    h,
                    context=text_emb,
                    cross_key_padding_mask=text_mask,
                    is_causal=True,
                )

            # Get last position logits
            h = self.norm(h)
            logits = self.output_proj(h[:, -1, :])  # [B, vocab_size]

            # Sample next token
            next_token = self._sample(logits, temperature, top_k, top_p)

            if self.length_control_mode == "stop_token" and self.stop_token_id is not None:
                # Append unless it's a pure stop for all samples
                if (next_token == self.stop_token_id).all():
                    break
                generated.append(next_token)
            else:
                generated.append(next_token)

                # Check for EOS or stop condition
                if eos_token is not None and (next_token == eos_token).all():
                    break
                if stop_condition is not None and stop_condition(len(generated)):
                    break

            # Append to sequence
            next_emb = self.audio_embedding(next_token.unsqueeze(1))
            x = torch.cat([x, next_emb], dim=1)

        if not generated:
            return torch.zeros(B, 0, dtype=torch.long, device=device)

        tokens = torch.stack(generated, dim=1)  # [B, L]

        if self.length_control_mode == "stop_token" and self.stop_token_id is not None:
            trimmed = []
            max_len = 0
            for b in range(B):
                row = tokens[b]
                stop_positions = (row == self.stop_token_id).nonzero(as_tuple=True)[0]
                if stop_positions.numel() > 0:
                    row = row[: stop_positions[0].item()]
                trimmed.append(row)
                max_len = max(max_len, row.numel())

            if max_len == 0:
                return torch.zeros(B, 0, dtype=torch.long, device=device)

            padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
            for b, row in enumerate(trimmed):
                if row.numel() == 0:
                    continue
                padded[b, : row.numel()] = row
                if row.numel() < max_len:
                    padded[b, row.numel() :] = row[-1]
            tokens = padded

        return tokens  # [B, L]

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample from logits with temperature and top-k/top-p filtering.

        Args:
            logits: Logits [B, vocab_size]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Sampled tokens [B]
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def compute_loss(
        self,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        prompt_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute cross-entropy loss.

        Args:
            audio_tokens: Target audio tokens [B, L]
            text_tokens: Text tokens [B, T]
            text_lengths: Text lengths [B]
            prompt_tokens: Speaker prompt tokens [B, P]
            prompt_lengths: Prompt lengths [B]
            audio_lengths: Actual audio lengths [B] (for masking)
            label_smoothing: Label smoothing factor

        Returns:
            Tuple of (loss, metrics dict)
        """
        # Forward pass
        logits = self.forward(
            audio_tokens, text_tokens, text_lengths,
            prompt_tokens, prompt_lengths
        )

        # Reshape for cross-entropy
        B, L, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = audio_tokens.reshape(-1)

        # Compute loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        loss = loss.reshape(B, L)

        # Mask padding if lengths provided
        if audio_lengths is not None:
            mask = torch.arange(L, device=audio_tokens.device).expand(B, L) < audio_lengths.unsqueeze(1)
            loss = (loss * mask.float()).sum() / mask.float().sum()
        else:
            loss = loss.mean()

        # Compute metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == audio_tokens).float().mean()
            perplexity = torch.exp(loss.detach())

        metrics = {
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item(),
        }

        return loss, metrics

    def get_interface_contract(self) -> dict:
        """Return checkpoint interface contract for AR/NAR compatibility checks."""
        return {
            "contract_version": 1,
            "model_type": "ar",
            "d_model": int(self.d_model),
            "vocab_size": int(self.codec_vocab_size),
            "text_vocab_size": int(self.text_vocab_size),
            "max_text_len": int(self.max_text_len),
            "max_seq_len": int(self.max_seq_len),
            "n_codebooks": 1,
            "length_control_mode": self.length_control_mode,
            "stop_token_enabled": self.stop_token_id is not None,
        }
