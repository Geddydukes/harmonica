"""LJSpeech dataset loader."""

import csv
from pathlib import Path
from typing import Optional

from .dataset import HarmonicaDataset, SpeechSample


class LJSpeechDataset(HarmonicaDataset):
    """LJSpeech dataset loader.

    LJSpeech is a single-speaker English speech dataset.
    ~24 hours of audio from a single female speaker.

    Dataset structure:
        LJSpeech-1.1/
        ├── wavs/
        │   ├── LJ001-0001.wav
        │   ├── LJ001-0002.wav
        │   └── ...
        └── metadata.csv

    metadata.csv format:
        LJ001-0001|text|normalized_text
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        min_duration: float = 0.5,
        max_duration: float = 10.0,
        sample_rate: int = 24000,
    ):
        """Initialize LJSpeech dataset.

        Args:
            data_dir: Path to LJSpeech-1.1 directory
            cache_dir: Path to cache directory
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            sample_rate: Target sample rate
        """
        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            sample_rate=sample_rate,
            validate_audio=True,
        )

    def _load_samples(self) -> None:
        """Load samples from metadata.csv."""
        metadata_path = self.data_dir / "metadata.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"LJSpeech metadata not found at {metadata_path}. "
                "Please download the dataset from https://keithito.com/LJ-Speech-Dataset/"
            )

        wavs_dir = self.data_dir / "wavs"

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) < 2:
                    continue

                file_id = row[0]
                # Use normalized text if available, otherwise original
                text = row[2] if len(row) > 2 else row[1]

                audio_path = wavs_dir / f"{file_id}.wav"
                if not audio_path.exists():
                    continue

                # Estimate duration from file size (rough)
                # LJSpeech is 22050 Hz, 16-bit mono
                # More accurate: use torchaudio.info
                file_size = audio_path.stat().st_size
                duration = (file_size - 44) / (22050 * 2)  # Approximate

                sample = SpeechSample(
                    audio_path=str(audio_path),
                    text=text,
                    speaker_id="ljspeech",  # Single speaker
                    duration=duration,
                )
                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from LJSpeech")

    def get_audio_duration(self, idx: int) -> float:
        """Get accurate duration using torchaudio."""
        import torchaudio

        sample = self.samples[idx]
        info = torchaudio.info(sample.audio_path)
        return info.num_frames / info.sample_rate
