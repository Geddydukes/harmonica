"""Typed-ish config validation and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class _PathSpec:
    path: tuple[str, ...]
    expected: type | tuple[type, ...]
    allow_none: bool = False


TOP_LEVEL_KEYS = {
    "model",
    "codec",
    "data",
    "training",
    "inference",
    "experiment",
    "device",
}


REQUIRED_PATHS: tuple[_PathSpec, ...] = (
    _PathSpec(("model",), dict),
    _PathSpec(("codec",), dict),
    _PathSpec(("data",), dict),
    _PathSpec(("training",), dict),
    _PathSpec(("experiment",), dict),
    _PathSpec(("device",), dict),
    _PathSpec(("model", "ar"), dict),
    _PathSpec(("model", "nar"), dict),
    _PathSpec(("codec", "sample_rate"), (int, float)),
    _PathSpec(("data", "max_audio_len"), (int, float)),
    _PathSpec(("data", "min_audio_len"), (int, float)),
    _PathSpec(("training", "batch_size"), int),
    _PathSpec(("training", "grad_accum_steps"), int),
    _PathSpec(("training", "lr"), (int, float)),
)


POSITIVE_PATHS: tuple[tuple[str, ...], ...] = (
    ("codec", "sample_rate"),
    ("data", "max_audio_len"),
    ("training", "batch_size"),
    ("training", "grad_accum_steps"),
    ("training", "lr"),
    ("training", "ema_update_every_updates"),
)


NON_NEGATIVE_PATHS: tuple[tuple[str, ...], ...] = (
    ("data", "min_audio_len"),
    ("data", "prompt_frames"),
    ("training", "warmup_steps"),
    ("training", "warmup_update_steps"),
    ("training", "max_steps"),
    ("training", "max_update_steps"),
    ("training", "token_noise_prob"),
    ("training", "codebook_entropy_weight"),
    ("training", "codebook_usage_entropy_weight"),
    ("training", "nar_conditioning_noise_prob"),
    ("training", "ema_decay"),
)


def _get(cfg: Dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _setdefault_nested(cfg: Dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = cfg
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur.setdefault(path[-1], value)


def _fmt_path(path: Iterable[str]) -> str:
    return ".".join(path)


def validate_config(config: Dict[str, Any], strict: bool = True, context: str = "train") -> None:
    """Validate high-level config integrity.

    This is intentionally lightweight: enough to catch silent config drift and
    typo mistakes without forcing heavy schema dependencies.
    """
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary")

    unknown_top = set(config.keys()) - TOP_LEVEL_KEYS
    if strict and unknown_top:
        unknown_str = ", ".join(sorted(unknown_top))
        raise ValueError(f"Unknown top-level config keys: {unknown_str}")
    if unknown_top:
        print(f"Warning: unknown top-level config keys ignored: {sorted(unknown_top)}")

    for spec in REQUIRED_PATHS:
        val = _get(config, spec.path)
        if val is None:
            raise ValueError(f"Missing required config path: {_fmt_path(spec.path)}")
        if val is None and spec.allow_none:
            continue
        if not isinstance(val, spec.expected):
            raise TypeError(
                f"Invalid type for {_fmt_path(spec.path)}: "
                f"expected {spec.expected}, got {type(val)}"
            )

    for path in POSITIVE_PATHS:
        val = _get(config, path)
        if val is None:
            continue
        if float(val) <= 0:
            raise ValueError(f"Expected {_fmt_path(path)} > 0, got {val}")

    for path in NON_NEGATIVE_PATHS:
        val = _get(config, path)
        if val is None:
            continue
        if float(val) < 0:
            raise ValueError(f"Expected {_fmt_path(path)} >= 0, got {val}")

    min_audio = float(_get(config, ("data", "min_audio_len")) or 0.0)
    max_audio = float(_get(config, ("data", "max_audio_len")) or 0.0)
    if max_audio and min_audio > max_audio:
        raise ValueError(
            "Invalid data duration bounds: "
            f"min_audio_len ({min_audio}) > max_audio_len ({max_audio})"
        )

    ema_decay = float(_get(config, ("training", "ema_decay")) or 0.0)
    if ema_decay < 0.0 or ema_decay >= 1.0:
        raise ValueError(f"training.ema_decay must be in [0, 1), got {ema_decay}")

    # Normalize missing optional defaults used by tooling/reporting.
    _setdefault_nested(config, ("data", "strict_cache_check"), False)
    _setdefault_nested(config, ("training", "use_ema"), False)
    _setdefault_nested(config, ("training", "ema_decay"), 0.999)
    _setdefault_nested(config, ("training", "ema_update_every_updates"), 1)
    _setdefault_nested(config, ("training", "eval_with_ema"), True)
    _setdefault_nested(config, ("training", "save_ema_in_checkpoints"), True)
    if context == "train":
        _setdefault_nested(config, ("data", "prompt_frames"), 0)


def compute_cache_fingerprint(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deterministic cache fingerprint from training-relevant fields."""
    return {
        "version": 1,
        "codec": {
            "type": config.get("codec", {}).get("type"),
            "bandwidth": float(config.get("codec", {}).get("bandwidth", 0.0)),
            "sample_rate": int(config.get("codec", {}).get("sample_rate", 0)),
        },
        "data": {
            "dataset": config.get("data", {}).get("dataset"),
            "min_audio_len": float(config.get("data", {}).get("min_audio_len", 0.0)),
            "max_audio_len": float(config.get("data", {}).get("max_audio_len", 0.0)),
        },
        "model": {
            "ar_max_seq_len": int(config.get("model", {}).get("ar", {}).get("max_seq_len", 0)),
            "nar_max_seq_len": int(config.get("model", {}).get("nar", {}).get("max_seq_len", 0)),
            "ar_vocab_size": int(config.get("model", {}).get("ar", {}).get("vocab_size", 0)),
            "nar_vocab_size": int(config.get("model", {}).get("nar", {}).get("vocab_size", 0)),
        },
    }


def diff_cache_fingerprint(
    expected: Dict[str, Any],
    cached: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Compare cache fingerprints.

    Returns:
        A tuple of:
        - mismatches: keys present in both fingerprints with different values.
        - missing: keys expected but absent in cached fingerprint.
    """

    mismatches: List[str] = []
    missing: List[str] = []

    def _walk(path: tuple[str, ...], exp: Any, got: Any) -> None:
        key = _fmt_path(path)
        if isinstance(exp, dict):
            if not isinstance(got, dict):
                mismatches.append(f"{key}: expected object, got {type(got).__name__}")
                return
            for child_key, child_exp in exp.items():
                child_path = path + (child_key,)
                if child_key not in got:
                    missing.append(_fmt_path(child_path))
                    continue
                _walk(child_path, child_exp, got[child_key])
            return
        if exp != got:
            mismatches.append(f"{key}: expected {exp!r}, got {got!r}")

    _walk(tuple(), expected, cached)
    return mismatches, missing
