"""Text normalization for TTS."""

import re
import unicodedata
from typing import Dict, Optional


class TextNormalizer:
    """Basic text normalization for TTS.

    Handles:
    - Unicode normalization
    - Lowercase conversion
    - Number expansion (basic)
    - Punctuation standardization
    - Whitespace normalization
    """

    # Common number words
    ONES = [
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    TENS = [
        "", "", "twenty", "thirty", "forty",
        "fifty", "sixty", "seventy", "eighty", "ninety",
    ]

    # Abbreviation expansions
    ABBREVIATIONS: Dict[str, str] = {
        "mr.": "mister",
        "mrs.": "missus",
        "ms.": "miss",
        "dr.": "doctor",
        "prof.": "professor",
        "st.": "saint",
        "jr.": "junior",
        "sr.": "senior",
        "vs.": "versus",
        "etc.": "etcetera",
        "e.g.": "for example",
        "i.e.": "that is",
    }

    def __init__(
        self,
        lowercase: bool = True,
        expand_numbers: bool = True,
        expand_abbreviations: bool = True,
    ):
        """Initialize normalizer.

        Args:
            lowercase: Convert to lowercase
            expand_numbers: Convert numbers to words
            expand_abbreviations: Expand common abbreviations
        """
        self.lowercase = lowercase
        self.expand_numbers = expand_numbers
        self.expand_abbreviations = expand_abbreviations

    def normalize(self, text: str) -> str:
        """Normalize text for TTS.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Unicode normalization (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Expand abbreviations
        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)

        # Expand numbers
        if self.expand_numbers:
            text = self._expand_numbers(text)

        # Standardize punctuation
        text = self._standardize_punctuation(text)

        # Normalize whitespace
        text = " ".join(text.split())

        return text.strip()

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbr, expansion in self.ABBREVIATIONS.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text

    def _expand_numbers(self, text: str) -> str:
        """Expand numbers to words (basic implementation)."""
        # Handle simple integers
        def replace_number(match):
            num = int(match.group(0))
            return self._number_to_words(num)

        # Replace standalone numbers (not part of larger patterns)
        text = re.sub(r"\b\d+\b", replace_number, text)

        return text

    def _number_to_words(self, n: int) -> str:
        """Convert integer to words (handles 0-999)."""
        if n < 0:
            return "negative " + self._number_to_words(-n)

        if n < 20:
            return self.ONES[n]

        if n < 100:
            tens, ones = divmod(n, 10)
            if ones == 0:
                return self.TENS[tens]
            return f"{self.TENS[tens]} {self.ONES[ones]}"

        if n < 1000:
            hundreds, remainder = divmod(n, 100)
            if remainder == 0:
                return f"{self.ONES[hundreds]} hundred"
            return f"{self.ONES[hundreds]} hundred {self._number_to_words(remainder)}"

        # For larger numbers, just spell out digits
        return " ".join(self.ONES[int(d)] for d in str(n))

    def _standardize_punctuation(self, text: str) -> str:
        """Standardize punctuation marks."""
        # Replace various quote marks with standard ones
        text = re.sub(r'[“”„"]', '"', text)
        text = re.sub(r"[‘’`']", "'", text)

        # Replace various dashes with standard hyphen
        text = re.sub(r"[–—−]", "-", text)

        # Remove or replace other special characters
        text = re.sub(r"…", "...", text)

        return text


def normalize_text(text: str, **kwargs) -> str:
    """Convenience function for text normalization.

    Args:
        text: Input text
        **kwargs: Arguments passed to TextNormalizer

    Returns:
        Normalized text
    """
    normalizer = TextNormalizer(**kwargs)
    return normalizer.normalize(text)
