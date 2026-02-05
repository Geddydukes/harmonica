#!/usr/bin/env python3
"""Synthesize speech from text using trained models."""

import argparse
from pathlib import Path

import torch

from harmonica.codec import EnCodecBackend
from harmonica.text import CharTokenizer
from harmonica.model import ARTransformer, NARTransformer
from harmonica.inference import Synthesizer
from harmonica.training.checkpoint import load_checkpoint
from harmonica.utils.device import get_device
from harmonica.utils.audio import save_audio


def main():
    parser = argparse.ArgumentParser(description="Synthesize speech from text")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output audio path",
    )
    parser.add_argument(
        "--ar-checkpoint",
        type=str,
        required=True,
        help="Path to AR model checkpoint",
    )
    parser.add_argument(
        "--nar-checkpoint",
        type=str,
        default=None,
        help="Path to NAR model checkpoint (optional)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )

    args = parser.parse_args()

    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load codec
    codec = EnCodecBackend(bandwidth=6.0, device=device)

    # Load AR model
    print(f"Loading AR model from {args.ar_checkpoint}")
    ar_ckpt = load_checkpoint(args.ar_checkpoint, device=device)
    ar_config = ar_ckpt["config"]["model"]["ar"]

    tokenizer = CharTokenizer()

    ar_model = ARTransformer(
        vocab_size=ar_config.get("vocab_size", 1024),
        text_vocab_size=ar_config.get("text_vocab_size", tokenizer.vocab_size),
        d_model=ar_config.get("d_model", 512),
        n_heads=ar_config.get("n_heads", 8),
        n_layers=ar_config.get("n_layers", 12),
        d_ff=ar_config.get("d_ff", 2048),
        dropout=ar_config.get("dropout", 0.1),
        max_seq_len=ar_config.get("max_seq_len", 2048),
        max_text_len=ar_config.get("max_text_len", 512),
        length_control_mode=ar_config.get("length_control_mode", "duration_predictor"),
        duration_hidden_dim=ar_config.get("duration_hidden_dim", 256),
    )
    load_checkpoint(args.ar_checkpoint, model=ar_model, device=device)

    # Load NAR model (optional)
    nar_model = None
    if args.nar_checkpoint:
        print(f"Loading NAR model from {args.nar_checkpoint}")
        nar_ckpt = load_checkpoint(args.nar_checkpoint, device=device)
        nar_config = nar_ckpt["config"]["model"]["nar"]

        nar_model = NARTransformer(
            n_codebooks=nar_config.get("n_codebooks", 7),
            vocab_size=nar_config.get("vocab_size", 1024),
            d_model=nar_config.get("d_model", 512),
            n_heads=nar_config.get("n_heads", 8),
            n_layers=nar_config.get("n_layers", 8),
            d_ff=nar_config.get("d_ff", 2048),
            dropout=nar_config.get("dropout", 0.1),
        )
        load_checkpoint(args.nar_checkpoint, model=nar_model, device=device)

    # Create synthesizer
    synthesizer = Synthesizer(
        ar_model=ar_model,
        nar_model=nar_model,
        codec=codec,
        tokenizer=tokenizer,
        device=device,
        ar_config={
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
    )

    # Synthesize
    print(f"Synthesizing: {args.text}")
    if args.reference:
        print(f"Using reference: {args.reference}")

    waveform = synthesizer.synthesize(
        text=args.text,
        reference_audio=args.reference,
    )

    # Save output
    save_audio(waveform, args.output, codec.sample_rate)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
