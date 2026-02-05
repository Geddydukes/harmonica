#!/usr/bin/env python3
"""Preprocess datasets: encode audio to codec tokens and cache."""

import argparse
from pathlib import Path
from typing import Optional
import json

import torch
from tqdm import tqdm

from harmonica.codec import EnCodecBackend
from harmonica.data import LJSpeechDataset, VCTKDataset, LibriTTSDataset
from harmonica.utils.audio import AudioPreprocessor, load_audio_file
from harmonica.utils.device import get_device


def preprocess_dataset(
    dataset,
    codec,
    device: torch.device,
    output_dir: Path,
    batch_size: int = 16,
) -> None:
    """Preprocess dataset by encoding audio to codec tokens.

    Args:
        dataset: Dataset to preprocess
        codec: Audio codec
        device: Computation device
        output_dir: Output directory for cached tokens
        batch_size: Batch size for encoding
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "samples": [],
        "codec": {
            "type": "encodec",
            "sample_rate": codec.sample_rate,
            "n_codebooks": codec.n_codebooks,
            "codebook_size": codec.codebook_size,
        },
    }

    print(f"Preprocessing {len(dataset)} samples...")

    preprocessor = AudioPreprocessor(target_sample_rate=codec.sample_rate)

    cache_idx = 0

    batch_waveforms = []
    batch_meta = []

    def flush_batch():
        nonlocal cache_idx, batch_waveforms, batch_meta
        if not batch_waveforms:
            return

        lengths = [w.shape[1] for w in batch_waveforms]
        max_len = max(lengths)

        padded = torch.zeros(len(batch_waveforms), 1, max_len, device=device)
        for i, w in enumerate(batch_waveforms):
            padded[i, :, : w.shape[1]] = w

        tokens = codec.encode(padded)  # [B, K, S]

        for i, meta in enumerate(batch_meta):
            n_frames = codec.samples_to_frames(lengths[i])
            sample_tokens = tokens[i, :, :n_frames].cpu()
            cache_path = output_dir / f"{cache_idx:08d}.pt"
            torch.save(sample_tokens, cache_path)
            metadata["samples"].append({
                "text": meta["text"],
                "speaker_id": meta["speaker_id"],
                "duration": meta["duration"],
                "cache_idx": cache_idx,
                "audio_path": meta["audio_path"],
            })
            cache_idx += 1

        batch_waveforms = []
        batch_meta = []

    for idx in tqdm(range(len(dataset))):
        sample = dataset.samples[idx]

        try:
            waveform, sr = load_audio_file(sample.audio_path)
            waveform, is_valid = preprocessor.preprocess(waveform, sr)
            if not is_valid:
                continue

            batch_waveforms.append(waveform.to(device))
            batch_meta.append({
                "text": sample.text,
                "speaker_id": sample.speaker_id,
                "duration": sample.duration,
                "audio_path": sample.audio_path,
            })

            if len(batch_waveforms) >= batch_size:
                flush_batch()

        except Exception as e:
            print(f"Error processing {sample.audio_path}: {e}")
            continue

    flush_batch()

    # Save metadata
    metadata_path = output_dir / "metadata.pt"
    torch.save(metadata, metadata_path)

    # Also save as JSON for inspection
    json_metadata = {
        "n_samples": len(metadata["samples"]),
        "codec": metadata["codec"],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(json_metadata, f, indent=2)

    print(f"Preprocessed {len(metadata['samples'])} samples")
    print(f"Cached to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess TTS dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ljspeech", "vctk", "libritts"],
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for cached tokens",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds",
    )

    args = parser.parse_args()

    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Setup codec
    codec = EnCodecBackend(bandwidth=6.0, device=device)

    # Load dataset
    data_dir = Path(args.data_dir)
    if args.dataset == "ljspeech":
        dataset = LJSpeechDataset(
            data_dir=str(data_dir),
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    elif args.dataset == "vctk":
        dataset = VCTKDataset(
            data_dir=str(data_dir),
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    elif args.dataset == "libritts":
        dataset = LibriTTSDataset(
            data_dir=str(data_dir),
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"./cache/{args.dataset}")

    # Preprocess
    preprocess_dataset(dataset, codec, device, output_dir)


if __name__ == "__main__":
    main()
