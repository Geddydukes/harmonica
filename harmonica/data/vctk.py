"""VCTK dataset loader."""

from pathlib import Path
from typing import Optional, Set
import os

from .dataset import HarmonicaDataset, SpeechSample


class VCTKDataset(HarmonicaDataset):
    """VCTK Corpus dataset loader.

    VCTK is a multi-speaker English speech dataset.
    ~44 hours of audio from 110 speakers with various accents.

    Dataset structure:
        VCTK-Corpus/
        ├── wav48/
        │   ├── p225/
        │   │   ├── p225_001.wav
        │   │   └── ...
        │   └── ...
        └── txt/
            ├── p225/
            │   ├── p225_001.txt
            │   └── ...
            └── ...

    Or newer version:
        VCTK-Corpus-0.92/
        ├── wav48_silence_trimmed/
        │   └── ...
        └── txt/
            └── ...
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        min_duration: float = 0.5,
        max_duration: float = 10.0,
        sample_rate: int = 24000,
        speakers: Optional[Set[str]] = None,
    ):
        """Initialize VCTK dataset.

        Args:
            data_dir: Path to VCTK-Corpus directory
            cache_dir: Path to cache directory
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            sample_rate: Target sample rate
            speakers: Set of speaker IDs to include (None = all)
        """
        self.speakers_filter = speakers
        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            sample_rate=sample_rate,
            validate_audio=True,
        )

    def _load_samples(self) -> None:
        """Load samples from VCTK structure."""
        # Find wav directory (varies by version)
        wav_dirs = ["wav48", "wav48_silence_trimmed"]
        wav_dir = None
        for d in wav_dirs:
            candidate = self.data_dir / d
            if candidate.exists():
                wav_dir = candidate
                break

        if wav_dir is None:
            raise FileNotFoundError(
                f"VCTK wav directory not found in {self.data_dir}. "
                "Please download from https://datashare.ed.ac.uk/handle/10283/3443"
            )

        txt_dir = self.data_dir / "txt"

        # Iterate through speaker directories
        for speaker_dir in sorted(wav_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name

            # Filter speakers if specified
            if self.speakers_filter and speaker_id not in self.speakers_filter:
                continue

            txt_speaker_dir = txt_dir / speaker_id

            # Iterate through audio files (wav/flac)
            audio_files = list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.flac"))
            for audio_file in sorted(audio_files):
                # Handle different naming conventions
                # Original: p225_001.wav
                # Silence trimmed: p225_001_mic1.flac or p225_001_mic2.flac
                stem = audio_file.stem
                if "_mic" in stem:
                    # Skip mic2, only use mic1
                    if "_mic2" in stem:
                        continue
                    base_name = stem.replace("_mic1", "")
                else:
                    base_name = stem

                # Find corresponding text file
                txt_path = txt_speaker_dir / f"{base_name}.txt"
                if not txt_path.exists():
                    continue

                # Read text
                try:
                    text = txt_path.read_text().strip()
                except Exception:
                    continue

                if not text:
                    continue

                # Estimate duration (prefer audio metadata; fallback to None)
                duration = None
                try:
                    import torchaudio

                    info = torchaudio.info(str(audio_file))
                    duration = info.num_frames / info.sample_rate
                except Exception:
                    duration = None

                sample = SpeechSample(
                    audio_path=str(audio_file),
                    text=text,
                    speaker_id=speaker_id,
                    duration=duration,
                )
                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from VCTK ({self.n_speakers} speakers)")

    def get_speaker_samples(self, speaker_id: str) -> list:
        """Get all samples for a specific speaker."""
        return [s for s in self.samples if s.speaker_id == speaker_id]
