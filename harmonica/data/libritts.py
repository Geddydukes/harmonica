"""LibriTTS dataset loader."""

from pathlib import Path
from typing import Optional, List
import os

from .dataset import HarmonicaDataset, SpeechSample


class LibriTTSDataset(HarmonicaDataset):
    """LibriTTS dataset loader.

    LibriTTS is a multi-speaker English TTS corpus derived from LibriSpeech.
    Available subsets: clean-100, clean-360, other-500

    Dataset structure:
        LibriTTS/
        ├── train-clean-100/
        │   ├── 103/
        │   │   ├── 1241/
        │   │   │   ├── 103_1241_000000_000000.wav
        │   │   │   ├── 103_1241_000000_000000.normalized.txt
        │   │   │   └── ...
        │   │   └── ...
        │   └── ...
        ├── train-clean-360/
        │   └── ...
        └── ...

    File naming: {speaker}_{chapter}_{utterance1}_{utterance2}.wav
    """

    SUBSETS = [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ]

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        subsets: Optional[List[str]] = None,
        min_duration: float = 0.5,
        max_duration: float = 15.0,
        sample_rate: int = 24000,
    ):
        """Initialize LibriTTS dataset.

        Args:
            data_dir: Path to LibriTTS directory
            cache_dir: Path to cache directory
            subsets: List of subsets to include (default: train-clean-360)
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            sample_rate: Target sample rate
        """
        self.subsets = subsets or ["train-clean-360"]
        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            sample_rate=sample_rate,
            validate_audio=True,
        )

    def _load_samples(self) -> None:
        """Load samples from LibriTTS structure."""
        for subset in self.subsets:
            subset_dir = self.data_dir / subset
            if not subset_dir.exists():
                print(f"Warning: LibriTTS subset {subset} not found at {subset_dir}")
                continue

            self._load_subset(subset_dir)

        print(f"Loaded {len(self.samples)} samples from LibriTTS ({self.n_speakers} speakers)")

    def _load_subset(self, subset_dir: Path) -> None:
        """Load samples from a single subset directory."""
        # Walk through speaker/chapter/files structure
        for speaker_dir in sorted(subset_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name

            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue

                # Find all wav files in this chapter
                for audio_file in chapter_dir.glob("*.wav"):
                    # Find corresponding normalized text
                    txt_path = audio_file.with_suffix(".normalized.txt")
                    if not txt_path.exists():
                        # Try original text
                        txt_path = audio_file.with_suffix(".original.txt")
                        if not txt_path.exists():
                            continue

                    try:
                        text = txt_path.read_text().strip()
                    except Exception:
                        continue

                    if not text:
                        continue

                    # Estimate duration
                    # LibriTTS is 24kHz, 16-bit mono
                    file_size = audio_file.stat().st_size
                    duration = (file_size - 44) / (24000 * 2)

                    sample = SpeechSample(
                        audio_path=str(audio_file),
                        text=text,
                        speaker_id=speaker_id,
                        duration=duration,
                    )
                    self.samples.append(sample)

    def get_speaker_chapters(self, speaker_id: str) -> List[str]:
        """Get list of chapters for a speaker."""
        chapters = set()
        for sample in self.samples:
            if sample.speaker_id == speaker_id:
                # Extract chapter from path
                path = Path(sample.audio_path)
                chapter = path.parent.name
                chapters.add(chapter)
        return sorted(list(chapters))


def download_libritts(
    output_dir: str,
    subsets: Optional[List[str]] = None,
) -> None:
    """Download LibriTTS dataset.

    Args:
        output_dir: Output directory
        subsets: Subsets to download (default: train-clean-100)
    """
    import urllib.request
    import tarfile

    subsets = subsets or ["train-clean-100"]
    base_url = "https://www.openslr.org/resources/60"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        tar_name = f"{subset}.tar.gz"
        url = f"{base_url}/{tar_name}"
        tar_path = output_dir / tar_name

        print(f"Downloading {subset}...")
        urllib.request.urlretrieve(url, tar_path)

        print(f"Extracting {subset}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(output_dir)

        # Clean up tar file
        tar_path.unlink()

    print("Done!")
