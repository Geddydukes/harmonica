"""Character-level tokenizer for text."""

from typing import Dict, List, Optional, Tuple

import torch

from .normalizer import TextNormalizer


class CharTokenizer:
    """Character-level tokenizer for TTS text input.

    Maps characters to integer indices. Includes special tokens:
    - PAD (0): Padding token
    - BOS (1): Beginning of sequence
    - EOS (2): End of sequence
    - UNK (3): Unknown character

    Default vocabulary includes:
    - Special tokens
    - Lowercase letters (a-z)
    - Digits (0-9)
    - Common punctuation
    - Space
    """

    # Default character set
    DEFAULT_CHARS = (
        " "  # Space
        "abcdefghijklmnopqrstuvwxyz"  # Letters
        "0123456789"  # Digits
        ".,!?;:'\"-"  # Punctuation
    )

    # Special token indices
    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(
        self,
        chars: Optional[str] = None,
        normalize: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        """Initialize tokenizer.

        Args:
            chars: Character vocabulary (excluding special tokens)
            normalize: Apply text normalization before tokenizing
            add_bos: Add BOS token to sequences
            add_eos: Add EOS token to sequences
        """
        self.chars = chars or self.DEFAULT_CHARS
        self.normalize = normalize
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Build vocabulary
        self._build_vocab()

        # Text normalizer
        if normalize:
            self.normalizer = TextNormalizer()
        else:
            self.normalizer = None

    def _build_vocab(self) -> None:
        """Build character to index mappings."""
        # Special tokens
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

        # Full vocabulary
        self.vocab = self.special_tokens + list(self.chars)

        # Mappings
        self.char_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char: Dict[int, str] = {i: c for i, c in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.vocab)

    @property
    def pad_idx(self) -> int:
        return self.PAD_IDX

    @property
    def bos_idx(self) -> int:
        return self.BOS_IDX

    @property
    def eos_idx(self) -> int:
        return self.EOS_IDX

    @property
    def unk_idx(self) -> int:
        return self.UNK_IDX

    def encode(self, text: str) -> List[int]:
        """Encode text to token indices.

        Args:
            text: Input text string

        Returns:
            List of token indices
        """
        # Normalize if enabled
        if self.normalizer is not None:
            text = self.normalizer.normalize(text)

        # Convert characters to indices
        tokens = []

        if self.add_bos:
            tokens.append(self.BOS_IDX)

        for char in text:
            idx = self.char_to_idx.get(char, self.UNK_IDX)
            tokens.append(idx)

        if self.add_eos:
            tokens.append(self.EOS_IDX)

        return tokens

    def decode(self, tokens: List[int], remove_special: bool = True) -> str:
        """Decode token indices to text.

        Args:
            tokens: List of token indices
            remove_special: Remove special tokens from output

        Returns:
            Decoded text string
        """
        chars = []
        for idx in tokens:
            if remove_special and idx in (self.PAD_IDX, self.BOS_IDX, self.EOS_IDX):
                continue
            char = self.idx_to_char.get(idx, "")
            if char not in self.special_tokens:
                chars.append(char)
        return "".join(chars)

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode batch of texts to padded tensor.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length (pad/truncate to this)
            padding: Pad sequences to same length

        Returns:
            Tuple of (tokens [B, L], lengths [B])
        """
        # Encode all texts
        encoded = [self.encode(text) for text in texts]
        lengths = [len(seq) for seq in encoded]

        # Determine max length
        if max_length is None:
            max_length = max(lengths)
        else:
            # Truncate if needed
            encoded = [seq[:max_length] for seq in encoded]
            lengths = [min(l, max_length) for l in lengths]

        # Pad sequences
        if padding:
            padded = []
            for seq in encoded:
                pad_len = max_length - len(seq)
                padded.append(seq + [self.PAD_IDX] * pad_len)
            tokens = torch.tensor(padded, dtype=torch.long)
        else:
            tokens = [torch.tensor(seq, dtype=torch.long) for seq in encoded]

        lengths = torch.tensor(lengths, dtype=torch.long)

        return tokens, lengths

    def __call__(self, text: str) -> List[int]:
        """Shorthand for encode."""
        return self.encode(text)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
