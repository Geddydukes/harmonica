"""Non-autoregressive Transformer for codebooks 2-8 prediction."""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import TokenEmbedding, SinusoidalPositionalEncoding
from .blocks import CrossAttentionBlock
from .text_encoder import TextEncoder
from .speaker import SpeakerEncoder


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
        text_vocab_size: int = 128,
        max_text_len: int = 512,
        text_padding_idx: int = 0,
        n_text_layers: int = 4,
        use_speaker_conditioning: bool = False,
        speaker_n_codebooks: int = 8,
        speaker_pooling: str = "mean",
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
            text_vocab_size: Text vocabulary size
            max_text_len: Maximum text sequence length
            text_padding_idx: Text padding token index
            n_text_layers: Number of text encoder layers
            use_speaker_conditioning: Enable speaker conditioning from reference tokens
            speaker_n_codebooks: Number of codebooks in speaker prompt tokens
            speaker_pooling: Speaker encoder pooling mode
        """
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.text_vocab_size = text_vocab_size
        self.use_speaker_conditioning = use_speaker_conditioning
        self._last_pred_tokens = None
        self._last_target_tokens = None

        # Text encoder for NAR conditioning.
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

        self.speaker_encoder = None
        if self.use_speaker_conditioning:
            self.speaker_encoder = SpeakerEncoder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_codebooks=speaker_n_codebooks,
                pooling=speaker_pooling,
            )

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

    def encode_text(
        self,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text tokens for NAR conditioning."""
        return self.text_encoder(text_tokens, text_lengths)

    def forward(
        self,
        ar_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
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
        if text_emb.shape[-1] != self.d_model:
            raise ValueError(
                f"NAR text embedding dim mismatch: got {text_emb.shape[-1]}, "
                f"expected {self.d_model}."
            )

        if codebook_idx is not None:
            return self._forward_single_codebook(
                ar_tokens, target_tokens, text_emb, text_mask, codebook_idx, speaker_emb
            )

        # Predict all codebooks
        all_logits = []
        for k in range(self.n_codebooks):
            logits = self._forward_single_codebook(
                ar_tokens, target_tokens, text_emb, text_mask, k, speaker_emb
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
        speaker_emb: Optional[torch.Tensor] = None,
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

        context = text_emb
        context_mask = text_mask
        if speaker_emb is not None:
            speaker_ctx = speaker_emb.unsqueeze(1)  # [B, 1, D]
            context = torch.cat([speaker_ctx, text_emb], dim=1)
            if text_mask is not None:
                spk_mask = torch.zeros(
                    (B, 1),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
                context_mask = torch.cat([spk_mask, text_mask], dim=1)
            else:
                context_mask = None

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
                context=context,
                cross_key_padding_mask=context_mask,
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
        speaker_tokens: Optional[torch.Tensor] = None,
        speaker_lengths: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate codebooks 2-8 given codebook 1.

        Args:
            ar_tokens: Codebook 1 tokens from AR model [B, L]
            text_emb: Text embeddings [B, T, D]
            text_mask: Text padding mask [B, T]
            temperature: Sampling temperature (0 = greedy)
            speaker_tokens: Optional reference codec tokens [B, K, L]
            speaker_lengths: Optional reference lengths [B]
            speaker_emb: Optional pre-computed speaker embeddings [B, D]

        Returns:
            All codebook tokens [B, K+1, L] (including codebook 1)
        """
        B, L = ar_tokens.shape
        device = ar_tokens.device

        if speaker_emb is None:
            speaker_emb = self._encode_speaker(speaker_tokens, speaker_lengths)

        # Start with AR tokens
        all_tokens = [ar_tokens]

        # Build up target tokens as we go
        generated = torch.zeros(B, self.n_codebooks, L, dtype=torch.long, device=device)

        for k in range(self.n_codebooks):
            # Forward for this codebook
            logits = self._forward_single_codebook(
                ar_tokens, generated, text_emb, text_mask, k, speaker_emb
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
        speaker_tokens: Optional[torch.Tensor] = None,
        speaker_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        entropy_weight: float = 0.0,
        usage_entropy_weight: float = 0.0,
        teacher_forcing_ratio: float = 1.0,
        conditioning_noise_prob: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute cross-entropy loss for all codebooks.

        Args:
            ar_tokens: Codebook 1 tokens [B, L]
            target_tokens: Target tokens for codebooks 2-8 [B, K, L]
            text_emb: Text embeddings [B, T, D]
            text_mask: Text padding mask [B, T]
            speaker_tokens: Optional speaker reference tokens [B, K, L]
            speaker_lengths: Optional speaker reference lengths [B]
            audio_lengths: Actual audio lengths [B]
            label_smoothing: Label smoothing factor
            entropy_weight: Optional entropy regularization weight
            usage_entropy_weight: Optional marginal-usage entropy regularization
            teacher_forcing_ratio: Probability of using ground-truth previous codebooks
            conditioning_noise_prob: Random replacement prob for conditioning tokens

        Returns:
            Tuple of (loss, metrics dict)
        """
        B, K, L = target_tokens.shape
        tf_ratio = float(max(0.0, min(1.0, teacher_forcing_ratio)))
        cond_noise_prob = float(max(0.0, min(1.0, conditioning_noise_prob)))
        device = target_tokens.device
        speaker_emb = self._encode_speaker(speaker_tokens, speaker_lengths)

        # Decode codebooks sequentially so conditioning can use scheduled sampling.
        conditioning_tokens = torch.zeros_like(target_tokens)
        logits_per_codebook = []
        preds_per_codebook = []
        total_loss = 0.0
        codebook_losses = []

        for k in range(K):
            k_logits = self._forward_single_codebook(
                ar_tokens=ar_tokens,
                target_tokens=conditioning_tokens,
                text_emb=text_emb,
                text_mask=text_mask,
                codebook_idx=k,
                speaker_emb=speaker_emb,
            )  # [B, L, V]
            logits_per_codebook.append(k_logits)

            k_logits_flat = k_logits.reshape(-1, self.vocab_size)
            k_targets = target_tokens[:, k, :].reshape(-1)

            k_loss = F.cross_entropy(
                k_logits_flat, k_targets,
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

            with torch.no_grad():
                k_preds = k_logits.argmax(dim=-1)
                preds_per_codebook.append(k_preds)
                if tf_ratio >= 1.0:
                    next_cond = target_tokens[:, k, :]
                elif tf_ratio <= 0.0:
                    next_cond = k_preds
                else:
                    gt_mask = torch.rand(B, L, device=device) < tf_ratio
                    next_cond = torch.where(gt_mask, target_tokens[:, k, :], k_preds)

                if cond_noise_prob > 0:
                    if audio_lengths is not None:
                        valid = (
                            torch.arange(L, device=device).unsqueeze(0)
                            < audio_lengths.unsqueeze(1)
                        )
                    else:
                        valid = torch.ones((B, L), device=device, dtype=torch.bool)
                    noise_mask = (torch.rand(B, L, device=device) < cond_noise_prob) & valid
                    if noise_mask.any():
                        rand_tokens = torch.randint(
                            low=0,
                            high=self.vocab_size,
                            size=(B, L),
                            device=device,
                            dtype=next_cond.dtype,
                        )
                        next_cond = torch.where(noise_mask, rand_tokens, next_cond)
                # Important: avoid in-place mutation on a tensor that was read
                # in the current forward graph; embedding backward on MPS tracks
                # index tensor versions and will error if they change in-place.
                updated_conditioning = conditioning_tokens.clone()
                updated_conditioning[:, k, :] = next_cond
                conditioning_tokens = updated_conditioning

        logits = torch.stack(logits_per_codebook, dim=1)  # [B, K, L, V]
        preds = torch.stack(preds_per_codebook, dim=1)  # [B, K, L]

        # Average across codebooks
        loss = total_loss / K

        entropy_loss = None
        if entropy_weight > 0:
            probs = torch.softmax(logits.float(), dim=-1)  # [B, K, L, V]
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B, K, L]
            if audio_lengths is not None:
                mask = (
                    torch.arange(L, device=ar_tokens.device).unsqueeze(0)
                    < audio_lengths.unsqueeze(1)
                )  # [B, L]
                mask = mask.unsqueeze(1).expand(B, K, L)
                entropy = (entropy * mask.float()).sum() / mask.float().sum()
            else:
                entropy = entropy.mean()
            entropy_loss = entropy_weight * (-entropy)
            loss = loss + entropy_loss

        usage_entropy_loss = None
        if usage_entropy_weight > 0:
            probs = torch.softmax(logits.float(), dim=-1)  # [B, K, L, V]
            if audio_lengths is not None:
                mask = (
                    torch.arange(L, device=ar_tokens.device).unsqueeze(0)
                    < audio_lengths.unsqueeze(1)
                )  # [B, L]
                mask = mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, L, 1]
                denom = mask.sum(dim=(0, 2)).clamp_min(1.0)  # [1, 1]
                marginal = (probs * mask).sum(dim=(0, 2)) / denom  # [K, V]
            else:
                marginal = probs.mean(dim=(0, 2))  # [K, V]

            usage_entropy = -(marginal * torch.log(marginal + 1e-8)).sum(dim=-1).mean()
            usage_entropy_loss = usage_entropy_weight * (-usage_entropy)
            loss = loss + usage_entropy_loss

        # Compute metrics
        with torch.no_grad():
            self._last_pred_tokens = preds.detach()
            self._last_target_tokens = target_tokens.detach()
            accuracy = (preds == target_tokens).float().mean()
            perplexity = torch.exp(loss.detach())

            # Marginal usage entropy diagnostics (pred vs target), averaged over codebooks.
            pred_usage_entropies = []
            target_usage_entropies = []
            for k in range(K):
                pred_k = preds[:, k, :]
                tgt_k = target_tokens[:, k, :]
                if audio_lengths is not None:
                    valid = (
                        torch.arange(L, device=device).unsqueeze(0)
                        < audio_lengths.unsqueeze(1)
                    )
                    pred_k = pred_k[valid]
                    tgt_k = tgt_k[valid]
                else:
                    pred_k = pred_k.reshape(-1)
                    tgt_k = tgt_k.reshape(-1)
                if pred_k.numel() == 0:
                    continue
                pred_hist = torch.bincount(pred_k, minlength=self.vocab_size).float()
                pred_probs = pred_hist / pred_hist.sum().clamp_min(1.0)
                pred_h = -(pred_probs * torch.log(pred_probs + 1e-8)).sum()
                pred_usage_entropies.append(pred_h)

                tgt_hist = torch.bincount(tgt_k, minlength=self.vocab_size).float()
                tgt_probs = tgt_hist / tgt_hist.sum().clamp_min(1.0)
                tgt_h = -(tgt_probs * torch.log(tgt_probs + 1e-8)).sum()
                target_usage_entropies.append(tgt_h)

            if pred_usage_entropies:
                pred_usage_entropy = torch.stack(pred_usage_entropies).mean()
            else:
                pred_usage_entropy = torch.tensor(0.0, device=device)
            if target_usage_entropies:
                target_usage_entropy = torch.stack(target_usage_entropies).mean()
            else:
                target_usage_entropy = torch.tensor(0.0, device=device)

        metrics = {
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item(),
        }
        if entropy_loss is not None:
            metrics["codebook_entropy_loss"] = entropy_loss.item()
        if usage_entropy_loss is not None:
            metrics["codebook_usage_entropy_loss"] = usage_entropy_loss.item()
        metrics["nar_teacher_forcing_ratio"] = tf_ratio
        metrics["nar_conditioning_noise_prob"] = cond_noise_prob
        metrics["pred_usage_entropy"] = pred_usage_entropy.item()
        metrics["target_usage_entropy"] = target_usage_entropy.item()
        metrics["nar_speaker_conditioning"] = 1.0 if speaker_emb is not None else 0.0
        for k, k_loss in enumerate(codebook_losses):
            metrics[f"loss_codebook_{k+2}"] = k_loss.item()

        return loss, metrics

    def get_interface_contract(self) -> dict:
        """Return checkpoint interface contract for AR/NAR compatibility checks."""
        return {
            "contract_version": 1,
            "model_type": "nar",
            "d_model": int(self.d_model),
            "vocab_size": int(self.vocab_size),
            "text_vocab_size": int(self.text_vocab_size),
            "max_text_len": int(self.max_text_len),
            "max_seq_len": int(self.max_seq_len),
            "n_codebooks": int(self.n_codebooks),
            "total_codebooks": int(self.n_codebooks + 1),
            "use_speaker_conditioning": bool(self.use_speaker_conditioning),
        }

    def _encode_speaker(
        self,
        speaker_tokens: Optional[torch.Tensor],
        speaker_lengths: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Encode speaker reference tokens if speaker conditioning is enabled."""
        if not self.use_speaker_conditioning or self.speaker_encoder is None:
            return None
        if speaker_tokens is None:
            return None

        ref_mask = None
        if speaker_lengths is not None:
            max_len = speaker_tokens.shape[-1]
            ref_mask = (
                torch.arange(max_len, device=speaker_tokens.device).unsqueeze(0)
                < speaker_lengths.unsqueeze(1)
            )
        return self.speaker_encoder(speaker_tokens, ref_mask=ref_mask)
