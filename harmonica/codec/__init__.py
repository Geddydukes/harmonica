"""Audio codec abstraction layer."""

from .interface import CodecInterface
from .encodec import EnCodecBackend

__all__ = [
    "CodecInterface",
    "EnCodecBackend",
]
