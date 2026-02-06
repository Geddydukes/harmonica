"""AR/NAR compatibility contracts."""

from __future__ import annotations

from typing import Dict, List, Optional


def build_ar_contract(ar_cfg: Dict, codec_cfg: Dict) -> Dict:
    """Build AR interface contract from config."""
    return {
        "contract_version": 1,
        "model_type": "ar",
        "d_model": int(ar_cfg.get("d_model", 0)),
        "vocab_size": int(ar_cfg.get("vocab_size", 0)),
        "text_vocab_size": int(ar_cfg.get("text_vocab_size", 0)),
        "max_text_len": int(ar_cfg.get("max_text_len", 0)),
        "max_seq_len": int(ar_cfg.get("max_seq_len", 0)),
        "codec_sample_rate": int(codec_cfg.get("sample_rate", 0)),
        "codec_bandwidth": float(codec_cfg.get("bandwidth", 0.0)),
    }


def build_nar_contract(nar_cfg: Dict, codec_cfg: Dict) -> Dict:
    """Build NAR interface contract from config."""
    return {
        "contract_version": 1,
        "model_type": "nar",
        "d_model": int(nar_cfg.get("d_model", 0)),
        "vocab_size": int(nar_cfg.get("vocab_size", 0)),
        "text_vocab_size": int(nar_cfg.get("text_vocab_size", 0)),
        "max_text_len": int(nar_cfg.get("max_text_len", 0)),
        "max_seq_len": int(nar_cfg.get("max_seq_len", 0)),
        "n_codebooks": int(nar_cfg.get("n_codebooks", 0)),
        "codec_sample_rate": int(codec_cfg.get("sample_rate", 0)),
        "codec_bandwidth": float(codec_cfg.get("bandwidth", 0.0)),
    }


def check_ar_nar_compatibility(
    ar_cfg: Dict,
    nar_cfg: Dict,
    codec_cfg: Optional[Dict] = None,
) -> List[str]:
    """Return a list of compatibility errors between AR and NAR interfaces."""
    errors: List[str] = []

    ar_type = str(ar_cfg.get("model_type", "ar"))
    nar_type = str(nar_cfg.get("model_type", "nar"))
    if ar_type != "ar":
        errors.append(f"AR contract model_type should be 'ar' (got {ar_type!r})")
    if nar_type != "nar":
        errors.append(f"NAR contract model_type should be 'nar' (got {nar_type!r})")

    ar_contract_version = int(ar_cfg.get("contract_version", 1))
    nar_contract_version = int(nar_cfg.get("contract_version", 1))
    if ar_contract_version != nar_contract_version:
        errors.append(
            "contract_version mismatch "
            f"(AR={ar_contract_version}, NAR={nar_contract_version})"
        )

    ar_d = int(ar_cfg.get("d_model", 0))
    nar_d = int(nar_cfg.get("d_model", 0))
    if ar_d != nar_d:
        errors.append(f"d_model mismatch (AR={ar_d}, NAR={nar_d})")

    ar_v = int(ar_cfg.get("vocab_size", 0))
    nar_v = int(nar_cfg.get("vocab_size", 0))
    if ar_v != nar_v:
        errors.append(f"vocab_size mismatch (AR={ar_v}, NAR={nar_v})")

    ar_tv = int(ar_cfg.get("text_vocab_size", 0))
    nar_tv = int(nar_cfg.get("text_vocab_size", 0))
    if ar_tv != nar_tv:
        errors.append(f"text_vocab_size mismatch (AR={ar_tv}, NAR={nar_tv})")

    ar_tmax = int(ar_cfg.get("max_text_len", 0))
    nar_tmax = int(nar_cfg.get("max_text_len", 0))
    if nar_tmax > ar_tmax and ar_tmax > 0:
        errors.append(f"max_text_len mismatch (AR={ar_tmax}, NAR={nar_tmax})")

    ar_smax = int(ar_cfg.get("max_seq_len", 0))
    nar_smax = int(nar_cfg.get("max_seq_len", 0))
    if nar_smax > ar_smax and ar_smax > 0:
        errors.append(f"max_seq_len mismatch (AR={ar_smax}, NAR={nar_smax})")

    n_codec = int((codec_cfg or {}).get("n_codebooks", 8))
    n_nar = int(nar_cfg.get("n_codebooks", 0))
    if n_nar != max(n_codec - 1, 0):
        errors.append(
            "n_codebooks mismatch "
            f"(NAR={n_nar}, expected codec_n_codebooks-1={max(n_codec-1, 0)})"
        )

    if "codec_sample_rate" in ar_cfg and "codec_sample_rate" in nar_cfg:
        ar_sr = int(ar_cfg.get("codec_sample_rate", 0))
        nar_sr = int(nar_cfg.get("codec_sample_rate", 0))
        if ar_sr != nar_sr:
            errors.append(f"codec_sample_rate mismatch (AR={ar_sr}, NAR={nar_sr})")

    if "codec_bandwidth" in ar_cfg and "codec_bandwidth" in nar_cfg:
        ar_bw = float(ar_cfg.get("codec_bandwidth", 0.0))
        nar_bw = float(nar_cfg.get("codec_bandwidth", 0.0))
        if abs(ar_bw - nar_bw) > 1e-6:
            errors.append(f"codec_bandwidth mismatch (AR={ar_bw}, NAR={nar_bw})")

    return errors
