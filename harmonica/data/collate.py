"""Batch collation for Harmonica datasets."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch

from ..text import CharTokenizer
from ..codec import CodecInterface
from ..utils.audio import AudioPreprocessor, load_audio_file


@dataclass
class HarmonicaBatch:
    """A batch of training samples."""

    # Audio codec tokens [B, K, L] where K = n_codebooks
    codec_tokens: torch.Tensor

    # Audio lengths (number of frames) [B]
    audio_lengths: torch.Tensor

    # Text tokens [B, T]
    text_tokens: torch.Tensor

    # Text lengths [B]
    text_lengths: torch.Tensor

    # Speaker IDs [B] (optional)
    speaker_ids: Optional[torch.Tensor] = None

    # Reference prompt tokens [B, K, P] (optional, for voice cloning)
    prompt_tokens: Optional[torch.Tensor] = None

    # Prompt lengths [B] (optional)
    prompt_lengths: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "HarmonicaBatch":
        """Move batch to device."""
        return HarmonicaBatch(
            codec_tokens=self.codec_tokens.to(device),
            audio_lengths=self.audio_lengths.to(device),
            text_tokens=self.text_tokens.to(device),
            text_lengths=self.text_lengths.to(device),
            speaker_ids=self.speaker_ids.to(device) if self.speaker_ids is not None else None,
            prompt_tokens=self.prompt_tokens.to(device) if self.prompt_tokens is not None else None,
            prompt_lengths=self.prompt_lengths.to(device) if self.prompt_lengths is not None else None,
        )

    def pin_memory(self) -> "HarmonicaBatch":
        """Pin memory for faster transfer to GPU."""
        return HarmonicaBatch(
            codec_tokens=self.codec_tokens.pin_memory(),
            audio_lengths=self.audio_lengths.pin_memory(),
            text_tokens=self.text_tokens.pin_memory(),
            text_lengths=self.text_lengths.pin_memory(),
            speaker_ids=self.speaker_ids.pin_memory() if self.speaker_ids is not None else None,
            prompt_tokens=self.prompt_tokens.pin_memory() if self.prompt_tokens is not None else None,
            prompt_lengths=self.prompt_lengths.pin_memory() if self.prompt_lengths is not None else None,
        )


class Collator:
    """Collates samples into batches with padding."""

    def __init__(
        self,
        tokenizer: CharTokenizer,
        codec: Optional[CodecInterface] = None,
        speaker_to_idx: Optional[Dict[str, int]] = None,
        prompt_frames: int = 0,
        sample_rate: int = 24000,
        preprocessor: Optional[AudioPreprocessor] = None,
        max_audio_len: Optional[int] = None,
    ):
        """Initialize collator.

        Args:
            tokenizer: Text tokenizer
            codec: Audio codec (for encoding audio on-the-fly if needed)
            speaker_to_idx: Mapping from speaker ID to index
            prompt_frames: Number of frames to use as speaker prompt (0 = no prompt)
            sample_rate: Audio sample rate
        """
        self.tokenizer = tokenizer
        self.codec = codec
        self.speaker_to_idx = speaker_to_idx or {}
        self.prompt_frames = prompt_frames
        self.sample_rate = sample_rate
        self.preprocessor = preprocessor or AudioPreprocessor(
            target_sample_rate=sample_rate
        )
        self.max_audio_len = max_audio_len

    def __call__(self, samples: List[Dict[str, Any]]) -> HarmonicaBatch:
        """Collate samples into a batch.

        Args:
            samples: List of sample dictionaries

        Returns:
            HarmonicaBatch
        """
        batch_size = len(samples)

        # Process codec tokens
        codec_tokens_list = []
        audio_lengths = []

        for sample in samples:
            if "codec_tokens" in sample and sample["codec_tokens"] is not None:
                tokens = sample["codec_tokens"]
            else:
                # Encode audio on-the-fly
                if self.codec is None:
                    raise ValueError("Codec required for on-the-fly encoding")

                if "audio_waveform" in sample and "audio_sample_rate" in sample:
                    waveform = sample["audio_waveform"]
                    sr = sample["audio_sample_rate"]
                else:
                    waveform, sr = load_audio_file(sample["audio_path"])
                waveform, is_valid = self.preprocessor.preprocess(waveform, sr)
                if not is_valid:
                    # Fallback to minimal zero tokens to keep batch intact
                    tokens = torch.zeros(
                        self.codec.n_codebooks, 1, dtype=torch.long
                    )
                else:
                    tokens = self.codec.encode(waveform.unsqueeze(0)).squeeze(0)

            if self.max_audio_len is not None and tokens.shape[-1] > self.max_audio_len:
                tokens = tokens[..., : self.max_audio_len]

            codec_tokens_list.append(tokens)
            audio_lengths.append(tokens.shape[-1])

        # Pad codec tokens
        max_audio_len = max(audio_lengths)
        n_codebooks = codec_tokens_list[0].shape[0]

        codec_tokens = torch.zeros(
            batch_size, n_codebooks, max_audio_len, dtype=torch.long
        )
        for i, tokens in enumerate(codec_tokens_list):
            codec_tokens[i, :, : tokens.shape[-1]] = tokens

        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        # Process text
        texts = [sample["text"] for sample in samples]
        text_tokens, text_lengths = self.tokenizer.encode_batch(texts)

        # Process speaker IDs
        speaker_ids = None
        if self.speaker_to_idx and any(s.get("speaker_id") for s in samples):
            speaker_ids = torch.tensor([
                self.speaker_to_idx.get(s.get("speaker_id", ""), 0)
                for s in samples
            ], dtype=torch.long)

        # Extract speaker prompts if needed
        prompt_tokens = None
        prompt_lengths = None

        if self.prompt_frames > 0:
            prompt_tokens_list = []
            prompt_lengths_list = []

            for tokens in codec_tokens_list:
                # Take first N frames as prompt
                prompt_len = min(self.prompt_frames, tokens.shape[-1])
                prompt = tokens[:, :prompt_len]
                prompt_tokens_list.append(prompt)
                prompt_lengths_list.append(prompt_len)

            # Pad prompts
            max_prompt_len = max(prompt_lengths_list)
            prompt_tokens = torch.zeros(
                batch_size, n_codebooks, max_prompt_len, dtype=torch.long
            )
            for i, prompt in enumerate(prompt_tokens_list):
                prompt_tokens[i, :, : prompt.shape[-1]] = prompt

            prompt_lengths = torch.tensor(prompt_lengths_list, dtype=torch.long)

        return HarmonicaBatch(
            codec_tokens=codec_tokens,
            audio_lengths=audio_lengths,
            text_tokens=text_tokens,
            text_lengths=text_lengths,
            speaker_ids=speaker_ids,
            prompt_tokens=prompt_tokens,
            prompt_lengths=prompt_lengths,
        )


def collate_fn(
    samples: List[Dict[str, Any]],
    tokenizer: CharTokenizer,
    codec: Optional[CodecInterface] = None,
    speaker_to_idx: Optional[Dict[str, int]] = None,
) -> HarmonicaBatch:
    """Functional interface for collation.

    Args:
        samples: List of sample dictionaries
        tokenizer: Text tokenizer
        codec: Audio codec
        speaker_to_idx: Speaker ID mapping

    Returns:
        HarmonicaBatch
    """
    collator = Collator(tokenizer, codec, speaker_to_idx)
    return collator(samples)
