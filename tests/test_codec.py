"""Tests for audio codec backends."""

import pytest
import torch

from harmonica.codec import CodecInterface, EnCodecBackend


class TestEnCodecBackend:
    """Tests for EnCodec backend."""

    @pytest.fixture
    def codec(self):
        """Create codec instance."""
        return EnCodecBackend(bandwidth=6.0, device=torch.device("cpu"))

    def test_properties(self, codec):
        """Test codec properties."""
        assert codec.sample_rate == 24000
        assert codec.n_codebooks == 8
        assert codec.codebook_size == 1024
        assert codec.hop_length == 320

    def test_encode_decode_shapes(self, codec):
        """Test encode/decode shape consistency."""
        # Create random audio (1 second)
        audio = torch.randn(1, 1, 24000)

        # Encode
        tokens = codec.encode(audio)
        assert tokens.dim() == 3
        assert tokens.shape[0] == 1  # Batch
        assert tokens.shape[1] == 8  # Codebooks
        assert tokens.shape[2] == 75  # Frames (24000 / 320)

        # Decode
        reconstructed = codec.decode(tokens)
        assert reconstructed.dim() == 3
        assert reconstructed.shape[0] == 1
        assert reconstructed.shape[1] == 1
        # Length should be close to original
        assert abs(reconstructed.shape[2] - 24000) < 320  # Within one hop

    def test_encode_2d_input(self, codec):
        """Test encoding with 2D input."""
        audio = torch.randn(2, 24000)  # [B, T]
        tokens = codec.encode(audio)
        assert tokens.shape[0] == 2
        assert tokens.shape[1] == 8

    def test_token_range(self, codec):
        """Test that tokens are in valid range."""
        audio = torch.randn(1, 1, 24000)
        tokens = codec.encode(audio)

        assert tokens.min() >= 0
        assert tokens.max() < codec.codebook_size

    def test_frames_to_samples(self, codec):
        """Test frame/sample conversion."""
        assert codec.frames_to_samples(75) == 24000
        assert codec.samples_to_frames(24000) == 75

    def test_reconstruct(self, codec):
        """Test reconstruction (encode + decode)."""
        audio = torch.randn(1, 1, 24000)
        reconstructed = codec.reconstruct(audio)

        assert reconstructed.shape[0] == 1
        assert reconstructed.shape[1] == 1

    def test_get_codebook_tokens(self, codec):
        """Test extracting tokens from specific codebook."""
        audio = torch.randn(1, 1, 24000)
        tokens = codec.encode(audio)

        cb0 = codec.get_codebook_tokens(tokens, 0)
        assert cb0.shape == (1, 75)

        cb7 = codec.get_codebook_tokens(tokens, 7)
        assert cb7.shape == (1, 75)


class TestCodecInterface:
    """Tests for codec interface methods."""

    @pytest.fixture
    def codec(self):
        return EnCodecBackend(bandwidth=6.0, device=torch.device("cpu"))

    def test_encode_to_flat(self, codec):
        """Test flattening codec tokens."""
        audio = torch.randn(1, 1, 24000)
        flat, seq_len = codec.encode_to_flat(audio)

        assert flat.dim() == 2
        assert flat.shape[0] == 1
        assert flat.shape[1] == seq_len * codec.n_codebooks
        assert seq_len == 75

    def test_decode_from_flat(self, codec):
        """Test decoding flattened tokens."""
        audio = torch.randn(1, 1, 24000)
        flat, seq_len = codec.encode_to_flat(audio)

        reconstructed = codec.decode_from_flat(flat, seq_len)
        assert reconstructed.dim() == 3
