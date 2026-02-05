"""Audio I/O utilities."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import soundfile as sf


class AudioPreprocessor:
    """Standardized audio preprocessing for Harmonica.

    All audio must pass through this before codec encoding to ensure
    consistent quality and format.
    """

    def __init__(
        self,
        target_sample_rate: int = 24000,
        normalization_method: str = "rms",
        target_db: float = -20.0,
        trim_silence: bool = True,
        silence_threshold_db: float = -40.0,
        trim_method: str = "both",
        min_duration: float = 0.5,
        max_duration: float = 30.0,
    ):
        """Initialize audio preprocessor.

        Args:
            target_sample_rate: Target sample rate (24kHz for EnCodec)
            normalization_method: 'rms', 'peak', or 'lufs'
            target_db: Target loudness in dB
            trim_silence: Whether to trim silence
            silence_threshold_db: Threshold for silence detection
            trim_method: 'start', 'end', or 'both'
            min_duration: Minimum valid clip duration (seconds)
            max_duration: Maximum valid clip duration (seconds)
        """
        self.target_sample_rate = target_sample_rate
        self.normalization_method = normalization_method
        self.target_db = target_db
        self.trim_silence = trim_silence
        self.silence_threshold_db = silence_threshold_db
        self.trim_method = trim_method
        self.min_duration = min_duration
        self.max_duration = max_duration

    def preprocess(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> Tuple[torch.Tensor, bool]:
        """Preprocess audio waveform.

        Args:
            waveform: Audio tensor [channels, samples] or [samples]
            sample_rate: Original sample rate

        Returns:
            Tuple of (preprocessed waveform [1, samples], is_valid)
        """
        # Ensure shape is [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to target rate
        if sample_rate != self.target_sample_rate:
            waveform = F.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )

        # Trim silence
        if self.trim_silence:
            waveform = self._trim_silence(waveform)

        # Check duration validity
        duration = waveform.shape[1] / self.target_sample_rate
        if duration < self.min_duration or duration > self.max_duration:
            return waveform, False

        # Normalize loudness
        waveform = self._normalize(waveform)

        # Check for NaN or Inf
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            return waveform, False

        return waveform, True

    def _trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Trim silence from start and/or end of audio."""
        threshold = 10 ** (self.silence_threshold_db / 20.0)

        # Compute frame energy (20ms frames)
        frame_length = int(0.02 * self.target_sample_rate)
        if waveform.shape[1] < frame_length:
            return waveform

        # Pad to multiple of frame_length
        pad_len = (frame_length - waveform.shape[1] % frame_length) % frame_length
        if pad_len > 0:
            padded = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            padded = waveform

        # Compute energy per frame
        frames = padded.view(1, -1, frame_length)
        energy = frames.abs().mean(dim=2).squeeze(0)

        # Find non-silent regions
        non_silent = energy > threshold

        if not non_silent.any():
            # Entire clip is silent - return small segment
            return waveform[:, :frame_length]

        non_silent_indices = non_silent.nonzero(as_tuple=True)[0]

        if self.trim_method in ["start", "both"]:
            start_idx = non_silent_indices[0].item() * frame_length
        else:
            start_idx = 0

        if self.trim_method in ["end", "both"]:
            end_idx = (non_silent_indices[-1].item() + 1) * frame_length
            end_idx = min(end_idx, waveform.shape[1])
        else:
            end_idx = waveform.shape[1]

        return waveform[:, start_idx:end_idx]

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to target loudness."""
        if self.normalization_method == "peak":
            peak = waveform.abs().max()
            if peak > 0:
                target_peak = 10 ** (self.target_db / 20.0)
                waveform = waveform * (target_peak / peak)

        elif self.normalization_method == "rms":
            rms = waveform.pow(2).mean().sqrt()
            if rms > 0:
                target_rms = 10 ** (self.target_db / 20.0)
                waveform = waveform * (target_rms / rms)

        elif self.normalization_method == "lufs":
            try:
                import pyloudnorm as pyln

                meter = pyln.Meter(self.target_sample_rate)
                audio_np = waveform.squeeze().numpy()
                loudness = meter.integrated_loudness(audio_np)
                if loudness > -70:  # Valid loudness
                    normalized = pyln.normalize.loudness(
                        audio_np, loudness, self.target_db
                    )
                    waveform = torch.from_numpy(normalized).unsqueeze(0)
            except ImportError:
                # Fallback to RMS
                rms = waveform.pow(2).mean().sqrt()
                if rms > 0:
                    target_rms = 10 ** (self.target_db / 20.0)
                    waveform = waveform * (target_rms / rms)

        # Prevent clipping
        waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform

    def preprocess_file(
        self,
        path: Union[str, Path],
    ) -> Tuple[torch.Tensor, bool]:
        """Preprocess audio file.

        Args:
            path: Path to audio file

        Returns:
            Tuple of (preprocessed waveform, is_valid)
        """
        waveform, sr = torchaudio.load(str(path))
        return self.preprocess(waveform, sr)


def load_audio(
    path: Union[str, Path],
    target_sr: int = 24000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load audio file and optionally resample.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default 24kHz for codec compatibility)
        mono: Convert to mono if True

    Returns:
        Tuple of (waveform tensor [C, T] or [T], sample_rate)
    """
    waveform, sr = torchaudio.load(str(path))

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)
        sr = target_sr

    # Remove channel dimension for mono
    if mono:
        waveform = waveform.squeeze(0)

    return waveform, sr


def load_audio_file(path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """Load audio file with torchaudio, falling back to soundfile.

    Args:
        path: Path to audio file

    Returns:
        Tuple of (waveform tensor [C, T], sample_rate)
    """
    try:
        return torchaudio.load(str(path))
    except Exception:
        data, sr = sf.read(str(path), always_2d=True)
        # soundfile returns [T, C]
        waveform = torch.from_numpy(data).transpose(0, 1).float()
        return waveform, sr


def save_audio(
    waveform: torch.Tensor,
    path: Union[str, Path],
    sample_rate: int = 24000,
) -> None:
    """Save waveform to audio file.

    Args:
        waveform: Audio tensor [T] or [C, T]
        path: Output path
        sample_rate: Sample rate
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure 2D tensor [C, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Move to CPU and ensure float32
    waveform = waveform.cpu().float()

    # Clamp to valid range
    waveform = torch.clamp(waveform, -1.0, 1.0)

    torchaudio.save(str(path), waveform, sample_rate)


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """Resample audio to target sample rate.

    Args:
        waveform: Audio tensor [C, T]
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled waveform
    """
    if orig_sr == target_sr:
        return waveform

    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform)


def get_audio_duration(
    path: Union[str, Path],
    sample_rate: Optional[int] = None,
) -> float:
    """Get audio duration in seconds.

    Args:
        path: Path to audio file
        sample_rate: If provided, calculate based on this rate

    Returns:
        Duration in seconds
    """
    info = torchaudio.info(str(path))
    if sample_rate is not None:
        return info.num_frames / sample_rate
    return info.num_frames / info.sample_rate


def trim_or_pad(
    waveform: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Trim or pad waveform to target length.

    Args:
        waveform: Audio tensor [T] or [C, T]
        target_length: Target number of samples
        pad_value: Value to use for padding

    Returns:
        Waveform with target length
    """
    current_length = waveform.shape[-1]

    if current_length > target_length:
        return waveform[..., :target_length]
    elif current_length < target_length:
        pad_amount = target_length - current_length
        if waveform.dim() == 1:
            padding = torch.full((pad_amount,), pad_value, device=waveform.device)
            return torch.cat([waveform, padding])
        else:
            padding = torch.full(
                (waveform.shape[0], pad_amount), pad_value, device=waveform.device
            )
            return torch.cat([waveform, padding], dim=-1)
    return waveform
