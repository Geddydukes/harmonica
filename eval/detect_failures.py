"""Detect common TTS failure modes."""

from typing import Dict
import os
import tempfile

import numpy as np
import torch
import torchaudio

try:
    import whisper
except Exception:
    whisper = None


class FailureModeDetector:
    """Detect common TTS failure modes automatically."""

    def __init__(self, sample_rate: int = 24000, enable_gibberish: bool = False):
        self.sample_rate = sample_rate
        self.enable_gibberish = enable_gibberish
        self.whisper_model = whisper.load_model("tiny") if (whisper and enable_gibberish) else None

    def detect_all_failures(
        self,
        audio: torch.Tensor,
        expected_text: str = None,
    ) -> Dict[str, bool]:
        """Run all failure detection checks."""
        return {
            "is_silent": self.is_silent(audio),
            "is_clipped": self.is_clipped(audio),
            "is_looping": self.is_looping(audio),
            "is_noisy": self.is_noisy(audio),
            "is_gibberish": self.is_gibberish(audio, expected_text) if (expected_text and self.enable_gibberish) else False,
        }

    def is_silent(self, audio: torch.Tensor, threshold_db: float = -50.0) -> bool:
        """Check if audio is effectively silent."""
        rms = audio.pow(2).mean().sqrt()
        rms_db = 20 * torch.log10(rms + 1e-8)
        return rms_db.item() < threshold_db

    def is_clipped(self, audio: torch.Tensor, threshold: float = 0.99) -> bool:
        """Check if audio is clipped."""
        peak = audio.abs().max()
        return peak > threshold

    def is_looping(self, audio: torch.Tensor, threshold: float = 0.8) -> bool:
        """Detect repeating patterns via autocorrelation."""
        audio_np = audio.squeeze().cpu().numpy()
        if audio_np.size == 0:
            return False

        autocorr = np.correlate(audio_np, audio_np, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        if autocorr[0] == 0:
            return False
        autocorr = autocorr / autocorr[0]

        min_lag = int(0.1 * self.sample_rate)
        if min_lag >= len(autocorr):
            return False
        max_correlation = autocorr[min_lag:].max()
        return max_correlation > threshold

    def is_noisy(self, audio: torch.Tensor, threshold: float = 0.7) -> bool:
        """Check if audio is mostly noise using spectral flatness."""
        spec = torch.stft(
            audio.squeeze(),
            n_fft=2048,
            hop_length=512,
            return_complex=True,
        )
        magnitude = spec.abs()

        geometric_mean = torch.exp(torch.log(magnitude + 1e-8).mean(dim=0))
        arithmetic_mean = magnitude.mean(dim=0)
        flatness = (geometric_mean / (arithmetic_mean + 1e-8)).mean()
        return flatness.item() > threshold

    def is_gibberish(
        self,
        audio: torch.Tensor,
        expected_text: str,
        wer_threshold: float = 0.8,
    ) -> bool:
        """Use Whisper to transcribe and compute WER."""
        if self.whisper_model is None:
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "temp_audio.wav")
            audio = audio.detach().cpu()
            torchaudio.save(temp_path, audio, self.sample_rate)

            result = self.whisper_model.transcribe(temp_path)
            transcription = result.get("text", "").strip().lower()
            expected = expected_text.strip().lower()

        wer = self._compute_wer(transcription, expected)
        return wer > wer_threshold

    def _compute_wer(self, hypothesis: str, reference: str) -> float:
        """Compute Word Error Rate."""
        hyp_words = hypothesis.split()
        ref_words = reference.split()

        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        for i in range(len(ref_words) + 1):
            d[i, 0] = i
        for j in range(len(hyp_words) + 1):
            d[0, j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min(
                        d[i - 1, j] + 1,
                        d[i, j - 1] + 1,
                        d[i - 1, j - 1] + 1,
                    )

        return d[len(ref_words), len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 1.0
