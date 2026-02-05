"""EnCodec backend implementation."""

from typing import Optional

import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

from .interface import CodecInterface


class EnCodecBackend(CodecInterface):
    """EnCodec neural audio codec backend.

    Uses the 24kHz EnCodec model with 8 codebooks.
    Each codebook has 1024 entries.
    Produces 75 frames per second (320 hop length at 24kHz).
    """

    def __init__(
        self,
        bandwidth: float = 6.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize EnCodec backend.

        Args:
            bandwidth: Target bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0)
                       Higher = more codebooks used = better quality
                       6.0 kbps uses all 8 codebooks at 24kHz
            device: Device to load model on
        """
        self.bandwidth = bandwidth
        self._device = device or torch.device("cpu")

        # Load 24kHz model
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(self._device)
        self.model.eval()

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Determine active codebooks for current bandwidth
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.sample_rate, device=self._device)
            codes = self.model.encode(dummy)[0][0]
            self._n_codebooks = int(codes.shape[1])

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def n_codebooks(self) -> int:
        return self._n_codebooks

    @property
    def codebook_size(self) -> int:
        return 1024

    @property
    def hop_length(self) -> int:
        # 24000 Hz / 75 fps = 320 samples per frame
        return 320

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> "EnCodecBackend":
        """Move model to device."""
        self._device = device
        self.model.to(device)
        return self

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to discrete tokens.

        Args:
            audio: Waveform tensor [B, T] or [B, 1, T]
                   Expected sample rate: 24000

        Returns:
            Discrete tokens [B, n_codebooks, S]
        """
        # Ensure correct shape [B, 1, T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        audio = audio.to(self._device)

        # Encode
        encoded_frames = self.model.encode(audio)

        # Extract codes from frames
        # EnCodec returns list of (codes, scale) tuples
        # codes shape: [B, n_codebooks, S]
        codes = encoded_frames[0][0]

        return codes

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens to audio waveform.

        Args:
            tokens: Discrete tokens [B, n_codebooks, S]

        Returns:
            Waveform tensor [B, 1, T]
        """
        tokens = tokens.to(self._device)

        # EnCodec expects list of (codes, scale) tuples
        encoded_frames = [(tokens, None)]

        # Decode
        audio = self.model.decode(encoded_frames)

        return audio

    def encode_file(self, path: str) -> torch.Tensor:
        """Convenience method to encode audio file directly.

        Args:
            path: Path to audio file

        Returns:
            Discrete tokens [1, n_codebooks, S]
        """
        import torchaudio

        waveform, sr = torchaudio.load(path)

        # Convert to mono and resample if needed
        waveform = convert_audio(
            waveform, sr, self.sample_rate, self.model.channels
        )

        # Add batch dimension
        waveform = waveform.unsqueeze(0)

        return self.encode(waveform)

    def reconstruct(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode then decode audio (for quality testing).

        Args:
            audio: Waveform tensor [B, T] or [B, 1, T]

        Returns:
            Reconstructed waveform [B, 1, T]
        """
        tokens = self.encode(audio)
        return self.decode(tokens)
