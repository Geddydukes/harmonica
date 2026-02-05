"""Tests for data loading and processing."""

import pytest
import torch
from unittest.mock import Mock, patch
from pathlib import Path

from harmonica.data.dataset import SpeechSample, HarmonicaDataset
from harmonica.data.collate import Collator, HarmonicaBatch
from harmonica.data.sampler import CurriculumSampler, LengthBatchSampler, BalancedSpeakerSampler
from harmonica.text import CharTokenizer


class TestSpeechSample:
    """Tests for SpeechSample dataclass."""

    def test_create_sample(self):
        """Test creating a speech sample."""
        sample = SpeechSample(
            audio_path="/path/to/audio.wav",
            text="Hello world",
            speaker_id="speaker1",
            duration=2.5,
        )

        assert sample.audio_path == "/path/to/audio.wav"
        assert sample.text == "Hello world"
        assert sample.speaker_id == "speaker1"
        assert sample.duration == 2.5
        assert sample.codec_tokens is None

    def test_sample_with_tokens(self):
        """Test sample with codec tokens."""
        tokens = torch.randint(0, 1024, (8, 50))
        sample = SpeechSample(
            audio_path="/path/to/audio.wav",
            text="Test",
            codec_tokens=tokens,
        )

        assert sample.codec_tokens is not None
        assert sample.codec_tokens.shape == (8, 50)


class TestCollator:
    """Tests for batch collation."""

    @pytest.fixture
    def tokenizer(self):
        return CharTokenizer()

    @pytest.fixture
    def mock_codec(self):
        codec = Mock()
        codec.sample_rate = 24000
        codec.n_codebooks = 8
        codec.encode.return_value = torch.randint(0, 1024, (1, 8, 75))
        return codec

    def test_collate_with_cached_tokens(self, tokenizer):
        """Test collation with pre-cached tokens."""
        samples = [
            {
                "text": "Hello world",
                "speaker_id": "spk1",
                "codec_tokens": torch.randint(0, 1024, (8, 75)),
                "idx": 0,
            },
            {
                "text": "Test sentence",
                "speaker_id": "spk2",
                "codec_tokens": torch.randint(0, 1024, (8, 100)),
                "idx": 1,
            },
        ]

        collator = Collator(tokenizer=tokenizer)
        batch = collator(samples)

        assert isinstance(batch, HarmonicaBatch)
        assert batch.codec_tokens.shape[0] == 2
        assert batch.codec_tokens.shape[1] == 8
        assert batch.text_tokens.shape[0] == 2
        assert batch.audio_lengths.shape == (2,)
        assert batch.text_lengths.shape == (2,)

    def test_collate_with_speaker_mapping(self, tokenizer):
        """Test collation with speaker ID mapping."""
        samples = [
            {
                "text": "Hello",
                "speaker_id": "spk1",
                "codec_tokens": torch.randint(0, 1024, (8, 75)),
                "idx": 0,
            },
            {
                "text": "World",
                "speaker_id": "spk2",
                "codec_tokens": torch.randint(0, 1024, (8, 75)),
                "idx": 1,
            },
        ]

        speaker_to_idx = {"spk1": 0, "spk2": 1}
        collator = Collator(tokenizer=tokenizer, speaker_to_idx=speaker_to_idx)
        batch = collator(samples)

        assert batch.speaker_ids is not None
        assert batch.speaker_ids.tolist() == [0, 1]

    def test_batch_to_device(self, tokenizer):
        """Test moving batch to device."""
        samples = [
            {
                "text": "Test",
                "codec_tokens": torch.randint(0, 1024, (8, 75)),
                "idx": 0,
            },
        ]

        collator = Collator(tokenizer=tokenizer)
        batch = collator(samples)

        # Move to CPU (should work on any machine)
        batch_cpu = batch.to(torch.device("cpu"))
        assert batch_cpu.codec_tokens.device.type == "cpu"


class TestCurriculumSampler:
    """Tests for curriculum sampler."""

    def test_basic_sampling(self):
        """Test basic sampling from multiple datasets."""
        # Mock datasets
        ds1 = list(range(100))
        ds2 = list(range(50))

        sampler = CurriculumSampler(
            datasets=[ds1, ds2],
            weights={"dataset_0": 0.5, "dataset_1": 0.5},
            total_samples=100,
            seed=42,
        )

        indices = list(sampler)
        assert len(indices) == 100

        # Indices should be in valid range
        for idx in indices:
            assert 0 <= idx < 150  # Total size

    def test_weight_adjustment(self):
        """Test adjusting weights."""
        ds1 = list(range(100))
        ds2 = list(range(100))

        sampler = CurriculumSampler(
            datasets=[ds1, ds2],
            total_samples=100,
        )

        # Update weights
        sampler.set_weights({"dataset_0": 0.9, "dataset_1": 0.1})

        # Sample and count
        indices = list(sampler)

        # Count samples from each dataset
        ds1_count = sum(1 for idx in indices if idx < 100)
        ds2_count = sum(1 for idx in indices if idx >= 100)

        # Should be roughly 90/10 split (with some variance)
        assert ds1_count > ds2_count


class TestLengthBatchSampler:
    """Tests for length-based batch sampler."""

    def test_batch_creation(self):
        """Test creating batches by length."""
        lengths = [10, 15, 12, 20, 18, 8, 25, 14]

        sampler = LengthBatchSampler(
            lengths=lengths,
            batch_size=2,
            shuffle=False,
        )

        batches = list(sampler)

        # Should have 4 batches of size 2
        assert len(batches) == 4
        for batch in batches:
            assert len(batch) == 2

    def test_drop_last(self):
        """Test dropping incomplete batch."""
        lengths = [10, 15, 12, 20, 18]  # 5 samples

        sampler = LengthBatchSampler(
            lengths=lengths,
            batch_size=2,
            drop_last=True,
        )

        batches = list(sampler)
        assert len(batches) == 2  # Only 2 complete batches


class TestBalancedSpeakerSampler:
    """Tests for balanced speaker sampler."""

    def test_balanced_sampling(self):
        """Test that speakers are balanced."""
        speaker_ids = ["spk1"] * 100 + ["spk2"] * 20 + ["spk3"] * 50

        sampler = BalancedSpeakerSampler(
            speaker_ids=speaker_ids,
            samples_per_speaker=30,
        )

        indices = list(sampler)

        # Should have 3 speakers * 30 samples
        assert len(indices) == 90

        # Count samples per speaker
        counts = {"spk1": 0, "spk2": 0, "spk3": 0}
        for idx in indices:
            if idx < 100:
                counts["spk1"] += 1
            elif idx < 120:
                counts["spk2"] += 1
            else:
                counts["spk3"] += 1

        # Each speaker should have 30 samples
        for count in counts.values():
            assert count == 30
