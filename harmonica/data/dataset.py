"""Base dataset class for Harmonica."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset
from ..utils.audio import AudioPreprocessor, load_audio_file


@dataclass
class SpeechSample:
    """A single speech sample."""

    audio_path: str
    text: str
    speaker_id: Optional[str] = None
    duration: Optional[float] = None

    # Cached codec tokens (set during preprocessing)
    codec_tokens: Optional[torch.Tensor] = None


class HarmonicaDataset(Dataset, ABC):
    """Abstract base class for TTS datasets.

    Subclasses implement dataset-specific loading logic.
    Common functionality:
    - Filtering by duration
    - Caching of codec tokens
    - Text normalization
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        min_duration: float = 0.5,
        max_duration: float = 15.0,
        sample_rate: int = 24000,
        validate_audio: bool = False,
    ):
        """Initialize dataset.

        Args:
            data_dir: Path to dataset root
            cache_dir: Path to cache directory for preprocessed data
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            sample_rate: Target sample rate
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.validate_audio = validate_audio
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        # Load and filter samples
        self.samples: List[SpeechSample] = []
        self._load_samples()
        self._filter_samples()

    @abstractmethod
    def _load_samples(self) -> None:
        """Load samples from dataset. Implement in subclasses."""
        pass

    def _filter_samples(self) -> None:
        """Filter samples by duration."""
        if self.min_duration > 0 or self.max_duration < float("inf"):
            original_count = len(self.samples)
            self.samples = [
                s for s in self.samples
                if s.duration is None or (
                    self.min_duration <= s.duration <= self.max_duration
                )
            ]
            filtered = original_count - len(self.samples)
            if filtered > 0:
                print(f"Filtered {filtered} samples by duration")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample.

        Returns dict with:
            - audio_path: Path to audio file
            - text: Transcript text
            - speaker_id: Speaker identifier (if available)
            - codec_tokens: Cached tokens (if available)
        """
        sample = self.samples[idx]

        if self.validate_audio and not (
            self.cache_dir is not None and self._get_cache_path(idx).exists()
        ):
            # Try a few samples to avoid repeated invalid audio
            attempts = 0
            while attempts < 5:
                try:
                    waveform, sr = load_audio_file(sample.audio_path)
                    _, is_valid = self.preprocessor.preprocess(waveform, sr)
                    if is_valid:
                        break
                except Exception:
                    pass
                idx = (idx + 1) % len(self.samples)
                sample = self.samples[idx]
                attempts += 1
            if attempts >= 5:
                print(f"Warning: could not validate audio after retries: {sample.audio_path}")

        item = {
            "audio_path": sample.audio_path,
            "text": sample.text,
            "speaker_id": sample.speaker_id,
            "idx": idx,
        }

        # Load cached tokens if available
        if self.cache_dir is not None:
            cache_path = self._get_cache_path(idx)
            if cache_path.exists():
                item["codec_tokens"] = torch.load(cache_path)

        return item

    def _get_cache_path(self, idx: int) -> Path:
        """Get cache file path for sample."""
        return self.cache_dir / f"{idx:08d}.pt"

    def cache_tokens(self, idx: int, tokens: torch.Tensor) -> None:
        """Save codec tokens to cache."""
        if self.cache_dir is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_cache_path(idx)
        torch.save(tokens, cache_path)

    def is_cached(self, idx: int) -> bool:
        """Check if sample is cached."""
        if self.cache_dir is None:
            return False
        return self._get_cache_path(idx).exists()

    @property
    def speaker_ids(self) -> List[str]:
        """Get list of unique speaker IDs."""
        ids = set()
        for sample in self.samples:
            if sample.speaker_id is not None:
                ids.add(sample.speaker_id)
        return sorted(list(ids))

    @property
    def n_speakers(self) -> int:
        """Get number of unique speakers."""
        return len(self.speaker_ids)

    @property
    def is_multi_speaker(self) -> bool:
        """Check if dataset has multiple speakers."""
        return self.n_speakers > 1


class CachedDataset(Dataset):
    """Dataset that loads from preprocessed cache.

    Assumes all samples have been preprocessed and cached.
    Much faster than loading and encoding on-the-fly.
    """

    def __init__(
        self,
        cache_dir: str,
        metadata_path: str,
    ):
        """Initialize cached dataset.

        Args:
            cache_dir: Path to cache directory
            metadata_path: Path to metadata file
        """
        self.cache_dir = Path(cache_dir)
        self.metadata_path = Path(metadata_path)

        # Load metadata
        self.metadata = torch.load(self.metadata_path)
        self.samples = self.metadata["samples"]
        self._speaker_ids = sorted(
            {s.get("speaker_id") for s in self.samples if s.get("speaker_id") is not None}
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a cached sample."""
        sample = self.samples[idx]
        cache_idx = sample.get("cache_idx", idx)
        cache_path = self.cache_dir / f"{cache_idx:08d}.pt"

        item = {
            "text": sample["text"],
            "speaker_id": sample.get("speaker_id"),
            "codec_tokens": torch.load(cache_path),
            "idx": idx,
        }

        return item

    @property
    def speaker_ids(self) -> List[str]:
        """Get list of unique speaker IDs."""
        return self._speaker_ids

    @property
    def n_speakers(self) -> int:
        """Get number of unique speakers."""
        return len(self._speaker_ids)

    @property
    def is_multi_speaker(self) -> bool:
        """Check if dataset has multiple speakers."""
        return self.n_speakers > 1
