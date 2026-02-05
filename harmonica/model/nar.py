"""Non-autoregressive Transformer for codebooks 2-8 prediction."""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import TokenEmbedding, SinusoidalPositionalEncoding, CodebookEmbedding
from .blocks import CrossAttentionBlock


class NARTransformer(nn.Module):
    """Non-autoregressive Transformer for parallel codebook prediction.

    Predicts codebooks 2-8 in parallel, conditioned on:
    - Codebook 1 tokens from AR model
    - Text embeddings (via cross-attention)
    - Speaker embedding

    For each codebook k, the model receives:
    - All tokens from codebooks 1 to k-1
    - Masked tokens for codebook k (to be predicted)
    """

    def __init__(
        self,
        n_codebooks: int = 7,
        vocab_size: int = 1024,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        """Initialize NAR Transformer.

        Args:
            n_codebooks: Number of codebooks to predict (2-8, so 7 total)
            vocab_size: Codebook vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding for first codebook (input from AR)
        self.ar_embedding = TokenEmbedding(vocab_size, d_model)

        # Embeddings for codebooks 2-8 (both input and output)
        self.codebook_embeddings = nn.ModuleList([
            TokenEmbedding(vocab_size, d_model)
            for _ in range(n_codebooks)
        ])

        # Codebook index embedding (to indicate which codebook we're predicting)
        self.codebook_idx_embedding = nn.Embedding(n_codebooks, d_model)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, max_seq_len, dropout
        )

        # Transformer layers (bidirectional - no causal mask)
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Output projections (one per codebook)
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, vocab_size)
            for _ in range(n_codebooks)
        ])

        # Initialize
        for proj in self.output_projs:
            nn.init.zeros_(proj.bias)
            nn.init.normal_(proj.weight, std=d_model**-0.5)

    def forward(
        self,
        ar_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        codebook_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            ar_tokens: Codebook 1 tokens [B, L]
                Training: MUST be ground truth (prevents error coupling).
                Inference: use AR model predictions.
            target_tokens: Target tokens for codebooks 2-8 [B, K, L]
            text_emb: Text embeddings [B, T, D]
            text_mask: Text padding mask [B, T]
            codebook_idx: If provided, only predict this codebook (0-indexed for 2-8)
                         If None, predict all codebooks

        Returns:
            Logits [B, K, L, vocab_size] or [B, L, vocab_size] if codebook_idx given
        """
        B, L = ar_tokens.shape

        if codebook_idx is not None:
            return self._forward_single_codebook(
                ar_tokens, target_tokens, text_emb, text_mask, codebook_idx
            )

        # Predict all codebooks
        all_logits = []
        for k in range(self.n_codebooks):
            logits = self._forward_single_codebook(
                ar_tokens, target_tokens, text_emb, text_mask, k
            )
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)  # [B, K, L, vocab_size]

    def _forward_single_codebook(
        self,
        ar_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        codebook_idx: int,
    ) -> torch.Tensor:
        """Forward pass for a single codebook.

        Args:
            ar_tokens: Codebook 1 tokens [B, L]
            target_tokens: Target tokens for codebooks 2-8 [B, K, L]
            text_emb: Text embeddings [B, T, D]
            text_mask: Text padding mask [B, T]
            codebook_idx: Which codebook to predict (0 = codebook 2, etc.)

        Returns:
            Logits [B, L, vocab_size]
        """
        B, L = ar_tokens.shape

        # Start with AR embedding (codebook 1)
        x = self.ar_embedding(ar_tokens)  # [B, L, D]

        # Add embeddings from previous codebooks (2 to k)
        for k in range(codebook_idx):
            prev_emb = self.codebook_embeddings[k](target_tokens[:, k, :])
            x = x + prev_emb

        # Add codebook index embedding (broadcast across sequence)
        idx_emb = self.codebook_idx_embedding(
            torch.tensor([codebook_idx], device=ar_tokens.device)
        )
        x = x + idx_emb.unsqueeze(0)  # [B, L, D]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer layers (bidirectional)
        for layer in self.layers:
            x = layer(
                x,
                context=text_emb,
                cross_key_padding_mask=text_mask,
                is_causal=False,  # NAR is bidirectional
            )

        # Final norm
        x = self.norm(x)

        # Output projection for this codebook
        logits = self.output_projs[codebook_idx](x)  # [B, L, vocab_size]

        return logits

    @torch.no_grad()
    def generate(
        self,
        ar_tokens: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate codebooks 2-8 given codebook 1.

        Args:
            ar_tokens: Codebook 1 tokens from AR model [B, L]
            text_emb: Text embeddings [B, T, D]
            text_mask: Text padding mask [B, T]
            temperature: Sampling temperature (0 = greedy)

        Returns:
            All codebook tokens [B, K+1, L] (including codebook 1)
        """
        B, L = ar_tokens.shape
        device = ar_tokens.device

        # Start with AR tokens
        all_tokens = [ar_tokens]

        # Build up target tokens as we go
        generated = torch.zeros(B, self.n_codebooks, L, dtype=torch.long, device=device)

        for k in range(self.n_codebooks):
            # Forward for this codebook
            logits = self._forward_single_codebook(
                ar_tokens, generated, text_emb, text_mask, k
            )

            # Sample or take argmax
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                tokens = torch.multinomial(
                    probs.reshape(-1, self.vocab_size), num_samples=1
                ).reshape(B, L)
            else:
                tokens = logits.argmax(dim=-1)

            # Store for next codebook
            generated[:, k, :] = tokens
            all_tokens.append(tokens)

        # Stack all codebooks [B, K+1, L]
        return torch.stack(all_tokens, dim=1)

    def compute_loss(
        self,
        ar_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute cross-entropy loss for all codebooks.

        Args:
            ar_tokens: Codebook 1 tokens [B, L]
            target_tokens: Target tokens for codebooks 2-8 [B, K, L]
            text_emb: Text embeddings [B, T, D]
            text_mask: Text padding mask [B, T]
            audio_lengths: Actual audio lengths [B]
            label_smoothing: Label smoothing factor

        Returns:
            Tuple of (loss, metrics dict)
        """
        B, K, L = target_tokens.shape

        # Forward pass for all codebooks
        logits = self.forward(ar_tokens, target_tokens, text_emb, text_mask)

        # Compute loss for each codebook
        total_loss = 0.0
        codebook_losses = []

        for k in range(K):
            k_logits = logits[:, k, :, :].reshape(-1, self.vocab_size)
            k_targets = target_tokens[:, k, :].reshape(-1)

            k_loss = F.cross_entropy(
                k_logits, k_targets,
                label_smoothing=label_smoothing,
                reduction="none"
            ).reshape(B, L)

            # Mask padding
            if audio_lengths is not None:
                mask = torch.arange(L, device=ar_tokens.device).expand(B, L) < audio_lengths.unsqueeze(1)
                k_loss = (k_loss * mask.float()).sum() / mask.float().sum()
            else:
                k_loss = k_loss.mean()

            codebook_losses.append(k_loss)
            total_loss = total_loss + k_loss

        # Average across codebooks
        loss = total_loss / K

        # Compute metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # [B, K, L]
            accuracy = (preds == target_tokens).float().mean()
            perplexity = torch.exp(loss.detach())

        metrics = {
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item(),
        }
        for k, k_loss in enumerate(codebook_losses):
            metrics[f"loss_codebook_{k+2}"] = k_loss.item()

        return loss, metrics
