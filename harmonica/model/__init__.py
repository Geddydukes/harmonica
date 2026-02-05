"""Model architecture for Harmonica."""

from .embedding import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from .blocks import TransformerBlock, MultiHeadAttention
from .text_encoder import TextEncoder
from .speaker import SpeakerEncoder
from .ar import ARTransformer
from .nar import NARTransformer
from .duration_predictor import DurationPredictor

__all__ = [
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "TransformerBlock",
    "MultiHeadAttention",
    "TextEncoder",
    "SpeakerEncoder",
    "ARTransformer",
    "NARTransformer",
    "DurationPredictor",
]
