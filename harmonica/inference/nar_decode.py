"""NAR decoding for codebooks 2-8."""

from typing import Optional, List

import torch


class NARDecoder:
    """Decoder for NAR transformer (codebooks 2-8)."""

    def __init__(
        self,
        model,
        temperature: float = 1.0,
        temperatures_per_codebook: Optional[List[float]] = None,
    ):
        """Initialize NAR decoder.

        Args:
            model: NAR transformer model
            temperature: Global sampling temperature
            temperatures_per_codebook: Per-codebook temperatures [7]
        """
        self.model = model
        self.temperature = temperature
        self.temperatures_per_codebook = temperatures_per_codebook

    @torch.no_grad()
    def decode(
        self,
        ar_tokens: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate codebooks 2-8 given codebook 1.

        Args:
            ar_tokens: Codebook 1 tokens from AR model [B, L]
            text_emb: Text embeddings [B, T, D]
            text_mask: Text padding mask [B, T]

        Returns:
            All codebook tokens [B, K, L]
        """
        temp = self.temperature

        # Use per-codebook temperatures if provided
        if self.temperatures_per_codebook:
            return self._decode_with_per_codebook_temp(
                ar_tokens, text_emb, text_mask
            )

        return self.model.generate(
            ar_tokens=ar_tokens,
            text_emb=text_emb,
            text_mask=text_mask,
            temperature=temp,
        )

    @torch.no_grad()
    def _decode_with_per_codebook_temp(
        self,
        ar_tokens: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Generate with different temperature per codebook.

        Args:
            ar_tokens: Codebook 1 tokens [B, L]
            text_emb: Text embeddings [B, T, D]
            text_mask: Text padding mask [B, T]

        Returns:
            All codebook tokens [B, K, L]
        """
        B, L = ar_tokens.shape
        device = ar_tokens.device

        # Start with AR tokens
        all_tokens = [ar_tokens]

        # Build up target tokens
        generated = torch.zeros(
            B, self.model.n_codebooks, L, dtype=torch.long, device=device
        )

        for k in range(self.model.n_codebooks):
            # Get temperature for this codebook
            temp = self.temperatures_per_codebook[k] if k < len(self.temperatures_per_codebook) else self.temperature

            # Forward for this codebook
            logits = self.model._forward_single_codebook(
                ar_tokens, generated, text_emb, text_mask, k
            )

            # Sample
            if temp > 0:
                probs = torch.softmax(logits / temp, dim=-1)
                tokens = torch.multinomial(
                    probs.reshape(-1, self.model.vocab_size), num_samples=1
                ).reshape(B, L)
            else:
                tokens = logits.argmax(dim=-1)

            generated[:, k, :] = tokens
            all_tokens.append(tokens)

        return torch.stack(all_tokens, dim=1)

    @torch.no_grad()
    def decode_iterative(
        self,
        ar_tokens: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        n_iterations: int = 1,
    ) -> torch.Tensor:
        """Iterative refinement decoding.

        Generate all codebooks, then refine by re-predicting
        conditioned on previous predictions.

        Args:
            ar_tokens: Codebook 1 tokens [B, L]
            text_emb: Text embeddings [B, T, D]
            text_mask: Text padding mask [B, T]
            n_iterations: Number of refinement iterations

        Returns:
            All codebook tokens [B, K, L]
        """
        # Initial generation
        tokens = self.model.generate(
            ar_tokens=ar_tokens,
            text_emb=text_emb,
            text_mask=text_mask,
            temperature=self.temperature,
        )

        # Refinement iterations
        for _ in range(n_iterations - 1):
            # Re-predict each codebook conditioned on current predictions
            generated = tokens[:, 1:, :]  # Exclude AR tokens

            for k in range(self.model.n_codebooks):
                logits = self.model._forward_single_codebook(
                    ar_tokens, generated, text_emb, text_mask, k
                )

                if self.temperature > 0:
                    probs = torch.softmax(logits / self.temperature, dim=-1)
                    new_tokens = torch.multinomial(
                        probs.reshape(-1, self.model.vocab_size), num_samples=1
                    ).reshape(tokens.shape[0], tokens.shape[2])
                else:
                    new_tokens = logits.argmax(dim=-1)

                generated[:, k, :] = new_tokens

            # Update tokens
            tokens = torch.cat([ar_tokens.unsqueeze(1), generated], dim=1)

        return tokens
