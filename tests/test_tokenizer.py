"""Tests for text tokenizer."""

import pytest
import torch

from harmonica.text import CharTokenizer, TextNormalizer


class TestCharTokenizer:
    """Tests for character tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer instance."""
        return CharTokenizer()

    def test_vocab_size(self, tokenizer):
        """Test vocabulary size."""
        # 4 special tokens + characters
        assert tokenizer.vocab_size > 4
        assert len(tokenizer) == tokenizer.vocab_size

    def test_special_tokens(self, tokenizer):
        """Test special token indices."""
        assert tokenizer.pad_idx == 0
        assert tokenizer.bos_idx == 1
        assert tokenizer.eos_idx == 2
        assert tokenizer.unk_idx == 3

    def test_encode_simple(self, tokenizer):
        """Test basic encoding."""
        tokens = tokenizer.encode("hello")

        # Should have BOS + 5 chars + EOS
        assert len(tokens) == 7
        assert tokens[0] == tokenizer.bos_idx
        assert tokens[-1] == tokenizer.eos_idx

    def test_encode_decode_roundtrip(self, tokenizer):
        """Test encode/decode roundtrip."""
        text = "hello world"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded == text

    def test_encode_with_normalization(self):
        """Test encoding with normalization."""
        tokenizer = CharTokenizer(normalize=True)

        # Should lowercase
        tokens = tokenizer.encode("Hello")
        decoded = tokenizer.decode(tokens)
        assert decoded == "hello"

    def test_encode_without_normalization(self):
        """Test encoding without normalization."""
        tokenizer = CharTokenizer(normalize=False, add_bos=False, add_eos=False)

        # Should handle but may map to UNK
        tokens = tokenizer.encode("hello")
        assert len(tokens) == 5

    def test_unknown_character(self, tokenizer):
        """Test handling of unknown characters."""
        # Greek letter should map to UNK
        tokens = tokenizer.encode("αβγ")

        # After normalization, unknowns become UNK
        for t in tokens[1:-1]:  # Skip BOS/EOS
            assert t == tokenizer.unk_idx or t >= 4

    def test_encode_batch(self, tokenizer):
        """Test batch encoding."""
        texts = ["hello", "world", "test sentence"]
        tokens, lengths = tokenizer.encode_batch(texts)

        assert isinstance(tokens, torch.Tensor)
        assert isinstance(lengths, torch.Tensor)
        assert tokens.shape[0] == 3
        assert lengths.shape[0] == 3

        # All sequences padded to same length
        assert tokens.shape[1] == max(lengths).item()

    def test_encode_batch_with_max_length(self, tokenizer):
        """Test batch encoding with truncation."""
        texts = ["hello world this is a long sentence"]
        tokens, lengths = tokenizer.encode_batch(texts, max_length=10)

        assert tokens.shape[1] == 10
        assert lengths[0] == 10

    def test_callable(self, tokenizer):
        """Test __call__ shorthand."""
        tokens1 = tokenizer.encode("hello")
        tokens2 = tokenizer("hello")
        assert tokens1 == tokens2


class TestTextNormalizer:
    """Tests for text normalizer."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance."""
        return TextNormalizer()

    def test_lowercase(self, normalizer):
        """Test lowercase conversion."""
        assert normalizer.normalize("Hello World") == "hello world"

    def test_number_expansion(self, normalizer):
        """Test number to words conversion."""
        assert "one" in normalizer.normalize("1 apple")
        assert "twenty" in normalizer.normalize("20 items")

    def test_abbreviation_expansion(self, normalizer):
        """Test abbreviation expansion."""
        assert "doctor" in normalizer.normalize("Dr. Smith")
        assert "mister" in normalizer.normalize("Mr. Jones")

    def test_whitespace_normalization(self, normalizer):
        """Test whitespace normalization."""
        result = normalizer.normalize("hello    world  ")
        assert "  " not in result
        assert result == "hello world"

    def test_no_normalization(self):
        """Test with normalization disabled."""
        normalizer = TextNormalizer(
            lowercase=False,
            expand_numbers=False,
            expand_abbreviations=False,
        )
        result = normalizer.normalize("Dr. Smith has 5 cats")
        assert "Dr." in result
        assert "5" in result
