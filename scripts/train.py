#!/usr/bin/env python3
"""Train Harmonica models."""

import argparse
import sys
from pathlib import Path
from functools import partial
from typing import Optional
import yaml

import torch
from torch.utils.data import DataLoader, ConcatDataset

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from harmonica.model import ARTransformer, NARTransformer
from harmonica.codec import EnCodecBackend
from harmonica.text import CharTokenizer
from harmonica.data import (
    LJSpeechDataset,
    VCTKDataset,
    LibriTTSDataset,
    CachedDataset,
    HFVCTKStreamingDataset,
    HFVCTKDataset,
)
from harmonica.data.sampler import CurriculumSampler
from harmonica.data.collate import Collator, HarmonicaBatch
from harmonica.training import Trainer, NARTrainer
from harmonica.utils.device import get_device
from harmonica.utils.seed import set_seed


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def warn_cache_mismatch(config: dict) -> None:
    """Warn if cached preprocessing settings differ from current config."""
    data_cfg = config.get("data", {})
    cache_dir = Path(data_cfg.get("cache_dir", ""))
    if not cache_dir:
        return
    metadata_path = cache_dir / "metadata.pt"
    if not metadata_path.exists():
        return
    try:
        metadata = torch.load(metadata_path, map_location="cpu")
    except Exception:
        return
    preprocess = metadata.get("preprocess", {})
    cached_max = preprocess.get("max_duration")
    cfg_max = data_cfg.get("max_audio_len")
    if cached_max is not None and cfg_max is not None and cached_max > cfg_max:
        print(
            f"WARNING: Cache max_duration ({cached_max}s) > config max_audio_len ({cfg_max}s). "
            "Consider re-preprocessing to avoid truncation."
        )


def merge_configs(base: dict, override: dict) -> dict:
    """Merge two config dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def create_dataloaders(
    config: dict,
    tokenizer: CharTokenizer,
    codec: EnCodecBackend,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create data loader from config."""
    data_cfg = config.get("data", {})
    dataset_type = data_cfg.get("dataset", "ljspeech")
    data_dir = data_cfg.get("data_dir", "./data")
    cache_dir = data_cfg.get("cache_dir")

    dataset = None
    sampler = None

    # Streaming dataset (no cache)
    if dataset_type == "hf_vctk_stream":
        dataset = HFVCTKStreamingDataset(
            dataset_name=data_cfg.get("hf_dataset", "confit/vctk-full"),
            split=data_cfg.get("hf_split", "train"),
            shuffle=data_cfg.get("stream_shuffle", True),
            shuffle_buffer=data_cfg.get("stream_shuffle_buffer", 1000),
            seed=config.get("experiment", {}).get("seed", 42),
        )
    elif dataset_type == "hf_vctk":
        dataset = HFVCTKDataset(
            dataset_name=data_cfg.get("hf_dataset", "confit/vctk-full"),
            split=data_cfg.get("hf_split", "train"),
        )

    # Check for cached data (single-dataset path)
    if dataset is None and dataset_type != "mixed" and cache_dir and Path(cache_dir).exists():
        metadata_path = Path(cache_dir) / "metadata.pt"
        if metadata_path.exists():
            print(f"Loading cached data from {cache_dir}")
            dataset = CachedDataset(cache_dir, str(metadata_path))

    # Load raw dataset if no cache
    if dataset is None:
        if dataset_type == "ljspeech":
            dataset = LJSpeechDataset(
                data_dir=data_dir,
                cache_dir=cache_dir,
                min_duration=data_cfg.get("min_audio_len", 0.5),
                max_duration=data_cfg.get("max_audio_len", 10.0),
            )
        elif dataset_type == "vctk":
            dataset = VCTKDataset(
                data_dir=data_dir,
                cache_dir=cache_dir,
                min_duration=data_cfg.get("min_audio_len", 0.5),
                max_duration=data_cfg.get("max_audio_len", 10.0),
            )
        elif dataset_type == "libritts":
            dataset = LibriTTSDataset(
                data_dir=data_dir,
                cache_dir=cache_dir,
                subsets=[data_cfg.get("subset", "train-clean-360")],
                min_duration=data_cfg.get("min_audio_len", 0.5),
                max_duration=data_cfg.get("max_audio_len", 15.0),
            )
        elif dataset_type == "mixed":
            datasets = []
            cache_root = Path(cache_dir) if cache_dir else None
            data_root = Path(data_dir)

            def maybe_cached(name: str, builder):
                if cache_root:
                    cache_path = cache_root / name
                    metadata_path = cache_path / "metadata.pt"
                    if metadata_path.exists():
                        print(f"Loading cached data from {cache_path}")
                        return CachedDataset(str(cache_path), str(metadata_path))
                return builder()

            # LJSpeech
            ljspeech_dir = data_root / "LJSpeech-1.1"
            if ljspeech_dir.exists() or (cache_root and (cache_root / "ljspeech").exists()):
                datasets.append(
                    maybe_cached(
                        "ljspeech",
                        lambda: LJSpeechDataset(
                            data_dir=str(ljspeech_dir),
                            cache_dir=str(cache_root / "ljspeech") if cache_root else None,
                            min_duration=data_cfg.get("min_audio_len", 0.5),
                            max_duration=data_cfg.get("max_audio_len", 10.0),
                        ),
                    )
                )

            # VCTK
            vctk_dir = data_root / "VCTK-Corpus"
            if not vctk_dir.exists():
                vctk_dir = data_root / "VCTK-Corpus-0.92"
            if vctk_dir.exists() or (cache_root and (cache_root / "vctk").exists()):
                datasets.append(
                    maybe_cached(
                        "vctk",
                        lambda: VCTKDataset(
                            data_dir=str(vctk_dir),
                            cache_dir=str(cache_root / "vctk") if cache_root else None,
                            min_duration=data_cfg.get("min_audio_len", 0.5),
                            max_duration=data_cfg.get("max_audio_len", 10.0),
                        ),
                    )
                )

            # LibriTTS
            libritts_dir = data_root / "LibriTTS"
            subsets = data_cfg.get("subsets")
            if subsets is None:
                subset = data_cfg.get("subset", "train-clean-360")
                subsets = [subset] if isinstance(subset, str) else list(subset)
            if libritts_dir.exists() or (cache_root and (cache_root / "libritts").exists()):
                datasets.append(
                    maybe_cached(
                        "libritts",
                        lambda: LibriTTSDataset(
                            data_dir=str(libritts_dir),
                            cache_dir=str(cache_root / "libritts") if cache_root else None,
                            subsets=subsets,
                            min_duration=data_cfg.get("min_audio_len", 0.5),
                            max_duration=data_cfg.get("max_audio_len", 15.0),
                        ),
                    )
                )

            if not datasets:
                raise ValueError("Mixed dataset requested but no dataset sources were found.")

            dataset = ConcatDataset(datasets)

            # Optional curriculum weights
            curriculum = data_cfg.get("curriculum", {})
            weights_cfg = curriculum.get("phase3")
            if weights_cfg:
                weights = {}
                for idx, name in enumerate(["ljspeech", "vctk", "libritts"]):
                    if name in weights_cfg and idx < len(datasets):
                        weights[f"dataset_{idx}"] = float(weights_cfg[name])
                total_samples = data_cfg.get("total_samples", len(dataset))
                sampler = CurriculumSampler(
                    datasets=list(datasets),
                    weights=weights,
                    total_samples=total_samples,
                    seed=config.get("experiment", {}).get("seed", 42),
                )
        else:
            raise ValueError(f"Unknown dataset: {dataset_type}")

    # Create collator
    speaker_to_idx = None
    if isinstance(dataset, ConcatDataset):
        speaker_ids = []
        for ds in dataset.datasets:
            if hasattr(ds, "speaker_ids"):
                speaker_ids.extend(ds.speaker_ids)
        speaker_ids = sorted(set(speaker_ids))
        if len(speaker_ids) > 1:
            speaker_to_idx = {sid: i for i, sid in enumerate(speaker_ids)}
    elif hasattr(dataset, "speaker_ids") and dataset.is_multi_speaker:
        speaker_to_idx = {sid: i for i, sid in enumerate(dataset.speaker_ids)}

    max_seq_len = config.get("model", {}).get("ar", {}).get("max_seq_len")
    if max_seq_len is not None:
        max_audio_len = max(1, int(max_seq_len) - 1)
    else:
        max_audio_len = None

    collator = Collator(
        tokenizer=tokenizer,
        codec=codec,
        speaker_to_idx=speaker_to_idx,
        sample_rate=config.get("codec", {}).get("sample_rate", 24000),
        max_audio_len=max_audio_len,
    )

    # Create dataloaders
    train_cfg = config.get("training", {})
    is_streaming = dataset_type == "hf_vctk_stream"
    prefer = str(config.get("device", {}).get("prefer", "")).lower()
    pin_memory = not (prefer in {"mps", "auto"} or is_streaming)

    num_workers = data_cfg.get("num_workers", 4)
    if prefer in {"mps", "auto"}:
        num_workers = 0
        pin_memory = False
    if is_streaming and num_workers > 0:
        num_workers = 0

    val_loader = None
    val_split = float(data_cfg.get("val_split", 0.0))
    if is_streaming:
        val_split = 0.0

    if sampler is not None and val_split > 0:
        print("Warning: val_split ignored because a sampler is in use.")
        val_split = 0.0

    if val_split > 0 and hasattr(dataset, "__len__"):
        total_len = len(dataset)
        val_len = max(1, int(total_len * val_split))
        train_len = total_len - val_len
        generator = torch.Generator().manual_seed(
            config.get("experiment", {}).get("seed", 42)
        )
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_len, val_len], generator=generator
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=False if is_streaming else sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        drop_last=False if is_streaming else True,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 8),
            shuffle=False,
            sampler=None,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train Harmonica models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/baseline.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ar",
        choices=["ar", "nar"],
        help="Model type to train",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )

    args = parser.parse_args()

    # Load config
    base_config = load_config("configs/config.yaml")
    experiment_config = load_config(args.config)
    config = merge_configs(base_config, experiment_config)

    # Override from command line
    if args.device:
        config["device"] = {"prefer": args.device}
    if args.seed:
        config["experiment"]["seed"] = args.seed

    # Set seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Setup device
    device = get_device(config.get("device", {}).get("prefer", "auto"))
    print(f"Using device: {device}")

    # Setup codec
    codec_cfg = config.get("codec", {})
    codec = EnCodecBackend(
        bandwidth=codec_cfg.get("bandwidth", 6.0),
        device=device,
    )

    # Setup tokenizer
    tokenizer = CharTokenizer()

    # Create data loaders
    train_loader, val_loader = create_dataloaders(config, tokenizer, codec)
    warn_cache_mismatch(config)
    try:
        print(f"Training samples: {len(train_loader.dataset)}")
    except TypeError:
        print("Training samples: streaming (unknown length)")

    # Print key config
    print(
        "Config: "
        f"max_seq_len={config.get('model', {}).get('ar', {}).get('max_seq_len', 'n/a')}, "
        f"max_audio_len={config.get('data', {}).get('max_audio_len', 'n/a')}, "
        f"batch_size={config.get('training', {}).get('batch_size', 'n/a')}, "
        f"grad_accum_steps={config.get('training', {}).get('grad_accum_steps', 'n/a')}, "
        f"warmup_steps={config.get('training', {}).get('warmup_steps', 'n/a')}"
    )

    # Create model
    if args.model_type == "ar":
        model_cfg = config.get("model", {}).get("ar", {})
        model = ARTransformer(
            vocab_size=model_cfg.get("vocab_size", 1024),
            text_vocab_size=model_cfg.get("text_vocab_size", tokenizer.vocab_size),
            d_model=model_cfg.get("d_model", 512),
            n_heads=model_cfg.get("n_heads", 8),
            n_layers=model_cfg.get("n_layers", 12),
            d_ff=model_cfg.get("d_ff", 2048),
            dropout=model_cfg.get("dropout", 0.1),
            max_seq_len=model_cfg.get("max_seq_len", 2048),
            max_text_len=model_cfg.get("max_text_len", 512),
            length_control_mode=model_cfg.get("length_control_mode", "duration_predictor"),
            duration_hidden_dim=model_cfg.get("duration_hidden_dim", 256),
        )
        trainer_cls = Trainer
    else:
        model_cfg = config.get("model", {}).get("nar", {})
        model = NARTransformer(
            n_codebooks=model_cfg.get("n_codebooks", 7),
            vocab_size=model_cfg.get("vocab_size", 1024),
            d_model=model_cfg.get("d_model", 512),
            n_heads=model_cfg.get("n_heads", 8),
            n_layers=model_cfg.get("n_layers", 8),
            d_ff=model_cfg.get("d_ff", 2048),
            dropout=model_cfg.get("dropout", 0.1),
        )
        trainer_cls = NARTrainer

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} ({n_trainable:,} trainable)")

    # Create trainer
    trainer = trainer_cls(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Train
    print("Starting training...")
    summary = trainer.train(resume=args.resume)

    print("\nTraining complete!")
    print(f"Total time: {summary['total_time_str']}")
    print(f"Best metrics: {summary['best_metrics']}")


if __name__ == "__main__":
    main()
