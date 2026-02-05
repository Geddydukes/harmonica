"""DAC (Descript Audio Codec) backend implementation."""

from typing import Optional

import torch

from .interface import CodecInterface


class DACBackend(CodecInterface):
    """Descript Audio Codec (DAC) backend.

    Uses the 24kHz DAC model with configurable codebooks.
    Higher quality than EnCodec at similar bitrates.
    """

    def __init__(
        self,
        model_type: str = "24khz",
        device: Optional[torch.device] = None,
    ):
        """Initialize DAC backend.

        Args:
            model_type: Model variant - "16khz", "24khz", or "44khz"
            device: Device to load model on
        """
        self.model_type = model_type
        self._device = device or torch.device("cpu")

        # Lazy import to avoid dependency if not used
        try:
            import dac
            from dac.utils import load_model
        except ImportError:
            raise ImportError(
                "descript-audio-codec not installed. "
                "Install with: pip install descript-audio-codec"
            )

        # Load model
        model_path = dac.utils.download(model_type=model_type)
        self.model = load_model(model_path)
        self.model.to(self._device)
        self.model.eval()

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Store model info
        self._sample_rate = {"16khz": 16000, "24khz": 24000, "44khz": 44100}[model_type]
        self._n_codebooks = self.model.n_codebooks
        self._codebook_size = self.model.codebook_size

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def n_codebooks(self) -> int:
        return self._n_codebooks

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def hop_length(self) -> int:
        # DAC 24kHz: 512 samples per frame
        return self.model.hop_length

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> "DACBackend":
        """Move model to device."""
        self._device = device
        self.model.to(device)
        return self

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to discrete tokens.

        Args:
            audio: Waveform tensor [B, T] or [B, 1, T]

        Returns:
            Discrete tokens [B, n_codebooks, S]
        """
        # Ensure correct shape [B, 1, T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        audio = audio.to(self._device)

        # DAC preprocessing
        from dac import DACFile

        # Encode to continuous latents then quantize
        z, codes, latents, _, _ = self.model.encode(audio)

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

        # Convert codes to continuous representation
        z, _, _ = self.model.quantizer.from_codes(tokens)

        # Decode
        audio = self.model.decode(z)

        return audio

    def reconstruct(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode then decode audio (for quality testing).

        Args:
            audio: Waveform tensor [B, T] or [B, 1, T]

        Returns:
            Reconstructed waveform [B, 1, T]
        """
        tokens = self.encode(audio)
        return self.decode(tokens)
