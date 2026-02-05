"""Checkpoint save/load utilities for portable training."""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import os
import random
import sys
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: dict,
    path: str,
    scaler: Optional[Any] = None,
    metrics: Optional[dict] = None,
    random_state: Optional[dict] = None,
    sampler: Optional[Any] = None,
    training_time: Optional[float] = None,
) -> None:
    """Save training checkpoint.

    Checkpoints are portable between devices (MPS/CUDA/CPU).

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        step: Current training step
        config: Training configuration
        path: Output path
        scaler: Gradient scaler state (optional)
        metrics: Training metrics (optional)
        random_state: Random state for reproducibility (optional)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config,
        "metrics": metrics or {},
        # Reproducibility
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        # Dataset sampler state (if supported)
        "sampler_state": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
        # Metadata
        "training_time": training_time,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "device": str(next(model.parameters()).device),
        "timestamp": datetime.now().isoformat(),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None,
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if random_state is not None:
        checkpoint["random_state"] = random_state

    # Save to temporary file first, then rename (atomic save)
    temp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)

    # Also save config as JSON for easy inspection
    config_path = path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "step": step,
                "config": config,
                "metrics": metrics or {},
                "training_time": training_time,
                "device": checkpoint.get("device"),
                "timestamp": checkpoint.get("timestamp"),
            },
            f,
            indent=2,
        )


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    sampler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Handles device mapping for portability between MPS/CUDA/CPU.

    Args:
        path: Checkpoint path
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        scaler: Gradient scaler to load state into (optional)
        device: Target device (auto-detected if None)
        strict: Strict loading for model state dict

    Returns:
        Checkpoint dictionary with step, config, metrics
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Determine device for loading
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Load checkpoint with device mapping
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load model state
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Move optimizer state to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        if checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Load scaler state
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Restore sampler state if available
    if sampler is not None and checkpoint.get("sampler_state"):
        if hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(checkpoint["sampler_state"])

    # Restore RNG state
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"].cpu())
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    return {
        "step": checkpoint.get("step", 0),
        "config": checkpoint.get("config", {}),
        "metrics": checkpoint.get("metrics", {}),
        "random_state": checkpoint.get("random_state"),
        "training_time": checkpoint.get("training_time"),
    }


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    # Sort by step number
    def get_step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    checkpoints.sort(key=get_step)
    return checkpoints[-1]


def cleanup_checkpoints(
    checkpoint_dir: str,
    keep_last: int = 5,
    keep_best: bool = True,
) -> None:
    """Remove old checkpoints, keeping only recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep
        keep_best: Also keep the best checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if len(checkpoints) <= keep_last:
        return

    # Sort by step number
    def get_step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    checkpoints.sort(key=get_step)

    # Keep best checkpoint if it exists
    best_path = checkpoint_dir / "checkpoint_best.pt"
    to_keep = set()
    if keep_best and best_path.exists():
        to_keep.add(best_path)

    # Keep last N checkpoints
    for ckpt in checkpoints[-keep_last:]:
        to_keep.add(ckpt)

    # Remove old checkpoints
    for ckpt in checkpoints:
        if ckpt not in to_keep:
            ckpt.unlink()
            # Also remove JSON config if exists
            json_path = ckpt.with_suffix(".json")
            if json_path.exists():
                json_path.unlink()
