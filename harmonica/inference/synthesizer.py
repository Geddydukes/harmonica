"""End-to-end speech synthesis."""

from pathlib import Path
from typing import Optional, Union

import torch

from ..codec import CodecInterface
from ..text import CharTokenizer
from ..utils.audio import load_audio, save_audio
from .ar_decode import ARDecoder
from .nar_decode import NARDecoder


class Synthesizer:
    """End-to-end TTS synthesis pipeline.

    Combines:
    - Text tokenization
    - AR model (codebook 1)
    - NAR model (codebooks 2-8)
    - Codec decoder (tokens → audio)
    """

    def __init__(
        self,
        ar_model,
        nar_model,
        codec: CodecInterface,
        tokenizer: CharTokenizer,
        device: Optional[torch.device] = None,
        ar_config: Optional[dict] = None,
        nar_config: Optional[dict] = None,
    ):
        """Initialize synthesizer.

        Args:
            ar_model: Trained AR transformer
            nar_model: Trained NAR transformer
            codec: Audio codec for decoding
            tokenizer: Text tokenizer
            device: Computation device
            ar_config: AR decoding config
            nar_config: NAR decoding config
        """
        self.ar_model = ar_model
        self.nar_model = nar_model
        self.codec = codec
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")

        # Move models to device
        self.ar_model.to(self.device)
        self.ar_model.eval()

        if self.nar_model is not None:
            self.nar_model.to(self.device)
            self.nar_model.eval()

        # Decoders
        ar_config = ar_config or {}
        self.ar_decoder = ARDecoder(
            self.ar_model,
            max_length=ar_config.get("max_length", 2048),
            temperature=ar_config.get("temperature", 0.8),
            top_k=ar_config.get("top_k", 50),
            top_p=ar_config.get("top_p", 0.95),
        )

        nar_config = nar_config or {}
        if self.nar_model is not None:
            self.nar_decoder = NARDecoder(
                self.nar_model,
                temperature=nar_config.get("temperature", 0.5),
            )
        else:
            self.nar_decoder = None

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        reference_audio: Optional[Union[str, torch.Tensor]] = None,
        reference_tokens: Optional[torch.Tensor] = None,
        max_prompt_frames: int = 225,  # ~3 seconds at 75 fps
    ) -> torch.Tensor:
        """Synthesize speech from text.

        Args:
            text: Input text
            reference_audio: Path to reference audio or waveform tensor
            reference_tokens: Pre-computed reference codec tokens
            max_prompt_frames: Maximum frames to use from reference

        Returns:
            Generated waveform [T]
        """
        # Tokenize text
        text_tokens = self.tokenizer.encode(text)
        text_tokens = torch.tensor([text_tokens], device=self.device)
        text_lengths = torch.tensor([text_tokens.shape[1]], device=self.device)

        # Process reference for voice cloning
        prompt_tokens = None
        nar_speaker_tokens = None
        nar_speaker_lengths = None
        if reference_audio is not None:
            ref_tokens = self._encode_reference_tokens(reference_audio, max_prompt_frames)
            prompt_tokens = ref_tokens[:, 0, :]
            nar_speaker_tokens = ref_tokens
            nar_speaker_lengths = torch.tensor(
                [ref_tokens.shape[-1]] * ref_tokens.shape[0],
                dtype=torch.long,
                device=self.device,
            )
        elif reference_tokens is not None:
            if reference_tokens.dim() == 3:
                ref_tokens = reference_tokens[:, :, :max_prompt_frames].to(self.device)
                prompt_tokens = ref_tokens[:, 0, :]
                nar_speaker_tokens = ref_tokens
                nar_speaker_lengths = torch.tensor(
                    [ref_tokens.shape[-1]] * ref_tokens.shape[0],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                prompt_tokens = reference_tokens[:, :max_prompt_frames].to(self.device)

        # Generate codebook 1 tokens (AR)
        ar_tokens = self.ar_decoder.decode(
            text_tokens=text_tokens,
            text_lengths=text_lengths,
            prompt_tokens=prompt_tokens,
        )

        # Generate codebooks 2-8 (NAR)
        if self.nar_decoder is not None:
            # Get text embeddings for NAR
            if hasattr(self.nar_model, "encode_text"):
                text_emb, text_mask = self.nar_model.encode_text(
                    text_tokens, text_lengths
                )
            else:
                # Backward-compatible fallback for older NAR implementations.
                text_emb, text_mask = self.ar_model.text_encoder(
                    text_tokens, text_lengths
                )

            # Generate remaining codebooks
            all_tokens = self.nar_decoder.decode(
                ar_tokens=ar_tokens,
                text_emb=text_emb,
                text_mask=text_mask,
                speaker_tokens=nar_speaker_tokens,
                speaker_lengths=nar_speaker_lengths,
            )
        else:
            # AR-only mode (lower quality)
            all_tokens = ar_tokens.unsqueeze(1)
            # Pad with zeros for remaining codebooks
            B, _, L = all_tokens.shape
            padding = torch.zeros(
                B, self.codec.n_codebooks - 1, L,
                dtype=torch.long, device=self.device
            )
            all_tokens = torch.cat([all_tokens, padding], dim=1)

        # Decode to audio
        audio = self.codec.decode(all_tokens)

        return audio.squeeze(0).squeeze(0)  # [T]

    def _process_reference(
        self,
        reference: Union[str, torch.Tensor],
        max_frames: int,
    ) -> torch.Tensor:
        """Process reference audio for AR prompting (first codebook only).

        Args:
            reference: Path to audio file or waveform tensor
            max_frames: Maximum frames to extract

        Returns:
            Reference tokens [1, P] (first codebook only)
        """
        tokens = self._encode_reference_tokens(reference, max_frames)
        return tokens[:, 0, :]

    def _encode_reference_tokens(
        self,
        reference: Union[str, torch.Tensor],
        max_frames: int,
    ) -> torch.Tensor:
        """Encode reference audio and return codec tokens.

        Returns:
            Reference tokens [1, K, P]
        """
        if isinstance(reference, str):
            # Load audio file
            waveform, _ = load_audio(reference, self.codec.sample_rate)
            waveform = waveform.to(self.device)
        else:
            waveform = reference.to(self.device)

        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        # Encode to tokens
        tokens = self.codec.encode(waveform)  # [1, K, S]

        # Limit prompt length
        return tokens[:, :, :max_frames]

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        reference_audio: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Synthesize speech and save to file.

        Args:
            text: Input text
            output_path: Output audio path
            reference_audio: Path to reference audio for voice cloning
            **kwargs: Additional arguments to synthesize()
        """
        waveform = self.synthesize(text, reference_audio=reference_audio, **kwargs)
        save_audio(waveform, output_path, self.codec.sample_rate)

    def synthesize_batch(
        self,
        texts: list,
        reference_audio: Optional[str] = None,
    ) -> list:
        """Synthesize multiple texts.

        Args:
            texts: List of input texts
            reference_audio: Shared reference audio

        Returns:
            List of waveforms
        """
        # Process reference once if provided
        reference_tokens = None
        if reference_audio is not None:
            reference_tokens = self._encode_reference_tokens(reference_audio, 225)

        waveforms = []
        for text in texts:
            waveform = self.synthesize(
                text,
                reference_tokens=reference_tokens,
            )
            waveforms.append(waveform)

        return waveforms


def load_synthesizer(
    checkpoint_dir: str,
    codec: CodecInterface,
    device: Optional[torch.device] = None,
) -> Synthesizer:
    """Load synthesizer from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing AR and NAR checkpoints
        codec: Audio codec
        device: Computation device

    Returns:
        Configured Synthesizer
    """
    from ..training.checkpoint import load_checkpoint
    from ..model import ARTransformer, NARTransformer

    checkpoint_dir = Path(checkpoint_dir)

    # Load AR model
    ar_ckpt_path = checkpoint_dir / "ar_checkpoint_best.pt"
    ar_ckpt = load_checkpoint(str(ar_ckpt_path), device=device)
    ar_config = ar_ckpt["config"]["model"]["ar"]

    ar_model = ARTransformer(**ar_config)
    load_checkpoint(str(ar_ckpt_path), model=ar_model, device=device)

    # Load NAR model (optional)
    nar_model = None
    nar_ckpt_path = checkpoint_dir / "nar_checkpoint_best.pt"
    if nar_ckpt_path.exists():
        nar_ckpt = load_checkpoint(str(nar_ckpt_path), device=device)
        nar_config = nar_ckpt["config"]["model"]["nar"]

        nar_model = NARTransformer(**nar_config)
        load_checkpoint(str(nar_ckpt_path), model=nar_model, device=device)

    # Create tokenizer
    tokenizer = CharTokenizer()

    return Synthesizer(
        ar_model=ar_model,
        nar_model=nar_model,
        codec=codec,
        tokenizer=tokenizer,
        device=device,
    )
