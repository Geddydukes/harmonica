"""Tests for model components."""

import pytest
import torch

from harmonica.model import (
    ARTransformer,
    NARTransformer,
    TextEncoder,
    SpeakerEncoder,
    TransformerBlock,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
)
from harmonica.model.embedding import TokenEmbedding, CodebookEmbedding


class TestEmbeddings:
    """Tests for embedding layers."""

    def test_sinusoidal_positional_encoding(self):
        """Test sinusoidal positional encoding."""
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000)
        x = torch.randn(2, 100, 256)

        output = pe(x)
        assert output.shape == x.shape

    def test_learned_positional_encoding(self):
        """Test learned positional encoding."""
        pe = LearnedPositionalEncoding(d_model=256, max_len=1000)
        x = torch.randn(2, 100, 256)

        output = pe(x)
        assert output.shape == x.shape

    def test_token_embedding(self):
        """Test token embedding."""
        emb = TokenEmbedding(vocab_size=1000, d_model=256)
        tokens = torch.randint(0, 1000, (2, 50))

        output = emb(tokens)
        assert output.shape == (2, 50, 256)

    def test_codebook_embedding(self):
        """Test multi-codebook embedding."""
        emb = CodebookEmbedding(n_codebooks=8, vocab_size=1024, d_model=256)
        tokens = torch.randint(0, 1024, (2, 8, 50))

        output = emb(tokens)
        assert output.shape == (2, 50, 256)


class TestAttention:
    """Tests for attention mechanisms."""

    def test_multi_head_attention(self):
        """Test multi-head attention."""
        attn = MultiHeadAttention(d_model=256, n_heads=8)

        q = torch.randn(2, 50, 256)
        k = torch.randn(2, 60, 256)
        v = torch.randn(2, 60, 256)

        output = attn(q, k, v)
        assert output.shape == (2, 50, 256)

    def test_causal_attention(self):
        """Test causal masking in attention."""
        attn = MultiHeadAttention(d_model=256, n_heads=8)

        x = torch.randn(2, 50, 256)
        output = attn(x, x, x, is_causal=True)
        assert output.shape == x.shape

    def test_attention_with_padding_mask(self):
        """Test attention with key padding mask."""
        attn = MultiHeadAttention(d_model=256, n_heads=8)

        q = torch.randn(2, 50, 256)
        k = torch.randn(2, 60, 256)
        v = torch.randn(2, 60, 256)

        # Mask last 10 positions
        mask = torch.zeros(2, 60, dtype=torch.bool)
        mask[:, 50:] = True

        output = attn(q, k, v, key_padding_mask=mask)
        assert output.shape == (2, 50, 256)


class TestTransformerBlock:
    """Tests for transformer blocks."""

    def test_transformer_block(self):
        """Test basic transformer block."""
        block = TransformerBlock(
            d_model=256, n_heads=8, d_ff=1024, dropout=0.1
        )
        x = torch.randn(2, 50, 256)

        output = block(x)
        assert output.shape == x.shape

    def test_transformer_block_causal(self):
        """Test transformer block with causal mask."""
        block = TransformerBlock(
            d_model=256, n_heads=8, d_ff=1024, dropout=0.1
        )
        x = torch.randn(2, 50, 256)

        output = block(x, is_causal=True)
        assert output.shape == x.shape


class TestTextEncoder:
    """Tests for text encoder."""

    @pytest.fixture
    def encoder(self):
        return TextEncoder(
            vocab_size=100,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=512,
        )

    def test_forward(self, encoder):
        """Test text encoder forward pass."""
        tokens = torch.randint(0, 100, (2, 30))

        output, mask = encoder(tokens)
        assert output.shape == (2, 30, 256)
        assert mask.shape == (2, 30)

    def test_with_lengths(self, encoder):
        """Test text encoder with explicit lengths."""
        tokens = torch.randint(0, 100, (2, 30))
        lengths = torch.tensor([20, 25])

        output, mask = encoder(tokens, lengths)

        # Check mask is correct
        assert mask[0, 20:].all()  # Masked after length
        assert not mask[0, :20].any()  # Not masked before length


class TestSpeakerEncoder:
    """Tests for speaker encoder."""

    @pytest.fixture
    def encoder(self):
        return SpeakerEncoder(
            vocab_size=1024,
            d_model=256,
            n_codebooks=8,
            pooling="mean",
        )

    def test_forward_3d(self, encoder):
        """Test speaker encoder with 3D input."""
        tokens = torch.randint(0, 1024, (2, 8, 50))  # [B, K, L]

        output = encoder(tokens)
        assert output.shape == (2, 256)

    def test_forward_2d(self, encoder):
        """Test speaker encoder with 2D input."""
        tokens = torch.randint(0, 1024, (2, 50))  # [B, L]

        output = encoder(tokens)
        assert output.shape == (2, 256)

    def test_with_mask(self, encoder):
        """Test speaker encoder with mask."""
        tokens = torch.randint(0, 1024, (2, 8, 50))
        mask = torch.ones(2, 50, dtype=torch.bool)
        mask[:, 30:] = False  # Only first 30 frames valid

        output = encoder(tokens, mask)
        assert output.shape == (2, 256)


class TestARTransformer:
    """Tests for AR transformer."""

    @pytest.fixture
    def model(self):
        return ARTransformer(
            vocab_size=1024,
            text_vocab_size=100,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_seq_len=200,
            max_text_len=100,
        )

    def test_forward(self, model):
        """Test AR forward pass."""
        audio_tokens = torch.randint(0, 1024, (2, 50))
        text_tokens = torch.randint(0, 100, (2, 30))

        logits = model(audio_tokens, text_tokens)
        assert logits.shape == (2, 50, 1024)

    def test_forward_with_prompt(self, model):
        """Test AR forward with speaker prompt."""
        audio_tokens = torch.randint(0, 1024, (2, 50))
        text_tokens = torch.randint(0, 100, (2, 30))
        prompt_tokens = torch.randint(0, 1024, (2, 20))

        logits = model(audio_tokens, text_tokens, prompt_tokens=prompt_tokens)
        assert logits.shape == (2, 50, 1024)

    def test_compute_loss(self, model):
        """Test loss computation."""
        audio_tokens = torch.randint(0, 1024, (2, 50))
        text_tokens = torch.randint(0, 100, (2, 30))

        loss, metrics = model.compute_loss(audio_tokens, text_tokens)

        assert loss.dim() == 0  # Scalar
        assert "accuracy" in metrics
        assert "perplexity" in metrics

    def test_generate(self, model):
        """Test generation."""
        text_tokens = torch.randint(0, 100, (1, 30))

        generated = model.generate(
            text_tokens,
            max_length=20,
            temperature=1.0,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] <= 20

    def test_parameter_count(self, model):
        """Test model has expected parameter count."""
        n_params = sum(p.numel() for p in model.parameters())
        # Should be in reasonable range for small model
        assert n_params > 1_000_000
        assert n_params < 100_000_000


class TestNARTransformer:
    """Tests for NAR transformer."""

    @pytest.fixture
    def model(self):
        return NARTransformer(
            n_codebooks=7,
            vocab_size=1024,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=512,
        )

    def test_forward_single_codebook(self, model):
        """Test NAR forward for single codebook."""
        ar_tokens = torch.randint(0, 1024, (2, 50))
        target_tokens = torch.randint(0, 1024, (2, 7, 50))
        text_emb = torch.randn(2, 30, 256)

        logits = model(ar_tokens, target_tokens, text_emb, codebook_idx=0)
        assert logits.shape == (2, 50, 1024)

    def test_forward_all_codebooks(self, model):
        """Test NAR forward for all codebooks."""
        ar_tokens = torch.randint(0, 1024, (2, 50))
        target_tokens = torch.randint(0, 1024, (2, 7, 50))
        text_emb = torch.randn(2, 30, 256)

        logits = model(ar_tokens, target_tokens, text_emb)
        assert logits.shape == (2, 7, 50, 1024)

    def test_generate(self, model):
        """Test NAR generation."""
        ar_tokens = torch.randint(0, 1024, (1, 50))
        text_emb = torch.randn(1, 30, 256)

        all_tokens = model.generate(ar_tokens, text_emb)

        # Should have AR tokens + 7 codebooks
        assert all_tokens.shape == (1, 8, 50)

    def test_compute_loss(self, model):
        """Test loss computation."""
        ar_tokens = torch.randint(0, 1024, (2, 50))
        target_tokens = torch.randint(0, 1024, (2, 7, 50))
        text_emb = torch.randn(2, 30, 256)

        loss, metrics = model.compute_loss(ar_tokens, target_tokens, text_emb)

        assert loss.dim() == 0
        assert "accuracy" in metrics
