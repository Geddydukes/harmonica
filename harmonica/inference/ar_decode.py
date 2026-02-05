"""AR decoding strategies."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class ARDecoder:
    """Decoding strategies for AR transformer."""

    def __init__(
        self,
        model,
        max_length: int = 2048,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ):
        """Initialize AR decoder.

        Args:
            model: AR transformer model
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
        """
        self.model = model
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    @torch.no_grad()
    def decode(
        self,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        eos_token: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens using sampling.

        Args:
            text_tokens: Text tokens [B, T]
            text_lengths: Text lengths [B]
            prompt_tokens: Speaker prompt tokens [B, P]
            eos_token: End of sequence token

        Returns:
            Generated tokens [B, L]
        """
        return self.model.generate(
            text_tokens=text_tokens,
            text_lengths=text_lengths,
            prompt_tokens=prompt_tokens,
            max_length=self.max_length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            eos_token=eos_token,
        )

    @torch.no_grad()
    def decode_with_repetition_penalty(
        self,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        eos_token: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens with repetition penalty.

        Args:
            text_tokens: Text tokens [B, T]
            text_lengths: Text lengths [B]
            prompt_tokens: Speaker prompt tokens [B, P]
            eos_token: End of sequence token

        Returns:
            Generated tokens [B, L]
        """
        B = text_tokens.shape[0]
        device = text_tokens.device

        # Encode text once
        text_emb, text_mask = self.model.text_encoder(text_tokens, text_lengths)

        # Start with BOS
        bos = self.model.bos_token.expand(B, -1, -1)

        # Handle prompt
        if prompt_tokens is not None:
            prompt_emb = self.model.audio_embedding(prompt_tokens)
            x = torch.cat([prompt_emb, bos], dim=1)
        else:
            x = bos

        generated = []
        past_tokens = [set() for _ in range(B)]

        for _ in range(self.max_length):
            # Add positional encoding
            x_pos = self.model.pos_encoding(x)

            # Apply transformer layers
            h = x_pos
            for layer in self.model.layers:
                h = layer(
                    h,
                    context=text_emb,
                    cross_key_padding_mask=text_mask,
                    is_causal=True,
                )

            # Get last position logits
            h = self.model.norm(h)
            logits = self.model.output_proj(h[:, -1, :])  # [B, vocab_size]

            # Apply repetition penalty
            if self.repetition_penalty != 1.0:
                for b in range(B):
                    for token in past_tokens[b]:
                        logits[b, token] /= self.repetition_penalty

            # Sample next token
            next_token = self._sample(logits)
            generated.append(next_token)

            # Track for repetition penalty
            for b in range(B):
                past_tokens[b].add(int(next_token[b].item()))

            # Check for EOS
            if eos_token is not None and (next_token == eos_token).all():
                break

            # Append to sequence
            next_emb = self.model.audio_embedding(next_token.unsqueeze(1))
            x = torch.cat([x, next_emb], dim=1)

        return torch.stack(generated, dim=1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from logits with filtering.

        Args:
            logits: Logits [B, vocab_size]

        Returns:
            Sampled tokens [B]
        """
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Top-k filtering
        if self.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def greedy_decode(
    model,
    text_tokens: torch.Tensor,
    text_lengths: Optional[torch.Tensor] = None,
    prompt_tokens: Optional[torch.Tensor] = None,
    max_length: int = 2048,
    eos_token: Optional[int] = None,
) -> torch.Tensor:
    """Simple greedy decoding (argmax).

    Args:
        model: AR transformer model
        text_tokens: Text tokens [B, T]
        text_lengths: Text lengths [B]
        prompt_tokens: Speaker prompt tokens [B, P]
        max_length: Maximum generation length
        eos_token: End of sequence token

    Returns:
        Generated tokens [B, L]
    """
    return model.generate(
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        prompt_tokens=prompt_tokens,
        max_length=max_length,
        temperature=0.0,  # Greedy
        top_k=1,
        top_p=1.0,
        eos_token=eos_token,
    )
