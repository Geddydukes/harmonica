"""Tests for config schema and AR/NAR compatibility helpers."""

from copy import deepcopy

from harmonica.config import (
    validate_config,
    compute_cache_fingerprint,
    diff_cache_fingerprint,
)
from harmonica.model import ARTransformer, NARTransformer
from harmonica.utils import check_ar_nar_compatibility


def _base_config() -> dict:
    return {
        "model": {
            "ar": {
                "vocab_size": 1024,
                "text_vocab_size": 128,
                "d_model": 192,
                "max_seq_len": 1024,
                "max_text_len": 256,
            },
            "nar": {
                "n_codebooks": 7,
                "vocab_size": 1024,
                "text_vocab_size": 128,
                "d_model": 192,
                "max_seq_len": 1024,
                "max_text_len": 256,
            },
        },
        "codec": {
            "type": "encodec",
            "bandwidth": 6.0,
            "sample_rate": 24000,
        },
        "data": {
            "dataset": "ljspeech",
            "max_audio_len": 3.0,
            "min_audio_len": 0.5,
        },
        "training": {
            "batch_size": 1,
            "grad_accum_steps": 8,
            "lr": 6.0e-5,
        },
        "inference": {},
        "experiment": {},
        "device": {"prefer": "cpu"},
    }


def test_validate_config_applies_defaults():
    cfg = _base_config()
    validate_config(cfg, strict=True, context="train")

    assert cfg["data"]["strict_cache_check"] is False
    assert cfg["training"]["use_ema"] is False
    assert cfg["training"]["ema_decay"] == 0.999
    assert cfg["training"]["ema_update_every_updates"] == 1


def test_diff_cache_fingerprint_reports_mismatch_and_missing():
    cfg = _base_config()
    expected = compute_cache_fingerprint(cfg)
    cached = deepcopy(expected)
    cached["codec"]["sample_rate"] = 16000
    del cached["model"]["nar_max_seq_len"]

    mismatches, missing = diff_cache_fingerprint(expected, cached)

    assert any("codec.sample_rate" in item for item in mismatches)
    assert "model.nar_max_seq_len" in missing


def test_check_ar_nar_compatibility_passes_for_matching_contracts():
    ar = ARTransformer(
        vocab_size=1024,
        text_vocab_size=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=256,
        max_text_len=128,
    )
    nar = NARTransformer(
        n_codebooks=7,
        vocab_size=1024,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=256,
        text_vocab_size=128,
        max_text_len=128,
        n_text_layers=1,
    )

    errors = check_ar_nar_compatibility(
        ar.get_interface_contract(),
        nar.get_interface_contract(),
        {"n_codebooks": 8},
    )
    assert errors == []


def test_check_ar_nar_compatibility_detects_vocab_mismatch():
    ar_contract = {
        "contract_version": 1,
        "model_type": "ar",
        "d_model": 192,
        "vocab_size": 1024,
        "text_vocab_size": 128,
        "max_text_len": 256,
        "max_seq_len": 1024,
    }
    nar_contract = {
        "contract_version": 1,
        "model_type": "nar",
        "d_model": 192,
        "vocab_size": 2048,
        "text_vocab_size": 128,
        "max_text_len": 256,
        "max_seq_len": 1024,
        "n_codebooks": 7,
    }

    errors = check_ar_nar_compatibility(ar_contract, nar_contract, {"n_codebooks": 8})
    assert any("vocab_size mismatch" in err for err in errors)
