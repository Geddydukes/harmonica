#!/usr/bin/env python3
"""Quick cache/codebook health report for codec token diversity."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List

import torch


def _iter_token_paths(cache_dir: Path, max_samples: int | None, seed: int) -> List[Path]:
    metadata_path = cache_dir / "metadata.pt"
    if metadata_path.exists():
        metadata = torch.load(metadata_path, map_location="cpu")
        samples = metadata.get("samples", [])
        paths = []
        for sample in samples:
            cache_idx = sample.get("cache_idx")
            if cache_idx is None:
                continue
            token_path = cache_dir / f"{int(cache_idx):08d}.pt"
            if token_path.exists():
                paths.append(token_path)
    else:
        paths = sorted(
            p for p in cache_dir.glob("*.pt") if p.name not in {"metadata.pt"}
        )

    if max_samples is not None and len(paths) > max_samples:
        rng = random.Random(seed)
        paths = rng.sample(paths, k=max_samples)

    return paths


def _ensure_vocab(hist: torch.Tensor, needed_vocab: int) -> torch.Tensor:
    if needed_vocab <= hist.shape[0]:
        return hist
    out = torch.zeros(needed_vocab, dtype=hist.dtype)
    out[: hist.shape[0]] = hist
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect codebook token diversity in cache")
    parser.add_argument("--cache-dir", type=str, required=True, help="Path to cache directory")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of cached samples to inspect",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample subsampling",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
        help="Initial codec vocab size (auto-expands if larger ids are found)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    paths = _iter_token_paths(cache_dir, args.max_samples, args.seed)
    if not paths:
        raise RuntimeError(f"No cache token files found in {cache_dir}")

    # Infer codebook count from first sample.
    first_tokens = torch.load(paths[0], map_location="cpu")
    if first_tokens.dim() == 3 and first_tokens.shape[0] == 1:
        first_tokens = first_tokens.squeeze(0)
    if first_tokens.dim() != 2:
        raise RuntimeError(
            f"Expected token tensor shape [K, L], got {tuple(first_tokens.shape)} from {paths[0]}"
        )

    n_codebooks = first_tokens.shape[0]
    hists = [torch.zeros(int(args.vocab_size), dtype=torch.long) for _ in range(n_codebooks)]
    totals = [0 for _ in range(n_codebooks)]

    for token_path in paths:
        tokens = torch.load(token_path, map_location="cpu")
        if tokens.dim() == 3 and tokens.shape[0] == 1:
            tokens = tokens.squeeze(0)
        if tokens.dim() != 2:
            continue
        if tokens.shape[0] != n_codebooks:
            continue

        for k in range(n_codebooks):
            tok = tokens[k].reshape(-1).long()
            if tok.numel() == 0:
                continue
            needed_vocab = int(tok.max().item()) + 1
            hists[k] = _ensure_vocab(hists[k], needed_vocab)
            binc = torch.bincount(tok, minlength=hists[k].shape[0])
            hists[k] += binc
            totals[k] += int(tok.numel())

    rel_utils = []
    print(f"Cache dir: {cache_dir}")
    print(f"Samples analyzed: {len(paths)}")
    print(f"Codebooks detected: {n_codebooks}")
    print("")
    print("Per-codebook target utilization:")

    for k in range(n_codebooks):
        hist = hists[k]
        total = int(hist.sum().item())
        vocab = int(hist.shape[0])
        unique = int((hist > 0).sum().item())
        abs_util = unique / max(vocab, 1)
        rel_denom = max(1, min(vocab, total))
        rel_util = unique / rel_denom
        rel_utils.append(rel_util)

        probs = hist.float()
        probs = probs / probs.sum().clamp_min(1.0)
        entropy = float((-(probs * torch.log(probs + 1e-8)).sum()).item())
        max_entropy = math.log(max(vocab, 2))
        norm_entropy = entropy / max(max_entropy, 1e-8)

        print(
            f"  codebook_{k+1}: "
            f"tokens={total}, unique={unique}, vocab={vocab}, "
            f"abs_util={abs_util:.1%}, rel_util={rel_util:.1%}, "
            f"norm_entropy={norm_entropy:.3f}"
        )

    rel_utils_sorted = sorted(rel_utils)
    median_rel = rel_utils_sorted[len(rel_utils_sorted) // 2]
    print("")
    print(
        "Summary: "
        f"min_rel={min(rel_utils):.1%}, median_rel={median_rel:.1%}, max_rel={max(rel_utils):.1%}"
    )
    if median_rel < 0.25:
        print("Health verdict: LOW target diversity (cache/data issue likely).")
    else:
        print("Health verdict: target diversity looks healthy (training-side collapse likely).")


if __name__ == "__main__":
    main()
