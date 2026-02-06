#!/usr/bin/env python3
"""Evaluate trained models on test sentences."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from harmonica.codec import EnCodecBackend
from harmonica.config import validate_config
from harmonica.text import CharTokenizer
from harmonica.model import ARTransformer, NARTransformer
from harmonica.inference import Synthesizer
from harmonica.training.checkpoint import load_checkpoint
from harmonica.utils import (
    build_ar_contract,
    build_nar_contract,
    check_ar_nar_compatibility,
)
from harmonica.utils.device import get_device
from harmonica.utils.audio import save_audio
from eval.detect_failures import FailureModeDetector


# Default evaluation sentences
DEFAULT_SENTENCES = [
    # Short sentences
    "Hello, world.",
    "How are you today?",
    "The quick brown fox.",
    "This is a test.",
    "Nice to meet you.",
    # Medium sentences
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Artificial intelligence is transforming how we interact with technology.",
    "She sells seashells by the seashore on sunny summer afternoons.",
    "Please remember to bring your umbrella if it looks like rain.",
    "The library was quiet except for the soft rustling of pages.",
    # Long sentences
    "In the heart of the ancient forest, where sunlight barely penetrated the thick canopy of leaves, a small stream wound its way through moss-covered rocks and fallen trees.",
    "The scientist carefully documented her findings, knowing that this discovery could fundamentally change our understanding of the natural world and open new avenues for research.",
]


def _validate_checkpoint_config(cfg: dict, name: str) -> None:
    """Best-effort config validation for loaded checkpoints."""
    try:
        validate_config(cfg, strict=False, context="eval")
    except Exception as exc:
        print(f"Warning: {name} checkpoint config validation failed: {exc}")


def _resolve_contract(ckpt: dict, model_key: str, model_cfg: dict, codec_cfg: dict) -> dict:
    """Load contract from checkpoint metadata or rebuild from config."""
    contract = ckpt.get("interface_contract")
    if isinstance(contract, dict):
        return contract
    if model_key == "ar":
        return build_ar_contract(model_cfg, codec_cfg)
    return build_nar_contract(model_cfg, codec_cfg)


def evaluate_model(
    synthesizer: Synthesizer,
    sentences: List[str],
    reference_audio: str = None,
    output_dir: Path = None,
    enable_gibberish: bool = False,
) -> Dict:
    """Evaluate model on test sentences.

    Args:
        synthesizer: Synthesizer instance
        sentences: List of test sentences
        reference_audio: Optional reference audio for voice cloning
        output_dir: Directory to save generated audio

    Returns:
        Evaluation results dictionary
    """
    results = {
        "sentences": [],
        "errors": [],
        "statistics": {},
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    durations = []
    lengths = []

    failure_detector = FailureModeDetector(
        sample_rate=synthesizer.codec.sample_rate,
        enable_gibberish=enable_gibberish,
    )
    failure_counts = {
        "silent": 0,
        "clipped": 0,
        "looping": 0,
        "noisy": 0,
        "gibberish": 0,
    }

    for i, text in enumerate(tqdm(sentences, desc="Evaluating")):
        try:
            # Synthesize
            waveform = synthesizer.synthesize(
                text=text,
                reference_audio=reference_audio,
            )

            # Calculate duration
            duration = len(waveform) / synthesizer.codec.sample_rate

            result = {
                "idx": i,
                "text": text,
                "duration_sec": duration,
                "text_length": len(text),
                "samples": len(waveform),
            }

            # Failure detection
            failures = failure_detector.detect_all_failures(
                audio=waveform.unsqueeze(0),
                expected_text=text,
            )
            for key, is_failure in failures.items():
                if is_failure:
                    failure_counts[key.replace("is_", "")] += 1
                    result.setdefault("warnings", []).append(key)

            results["sentences"].append(result)
            durations.append(duration)
            lengths.append(len(text))

            # Save audio if output dir specified
            if output_dir:
                audio_path = output_dir / f"sentence_{i:03d}.wav"
                save_audio(waveform, str(audio_path), synthesizer.codec.sample_rate)

        except Exception as e:
            results["errors"].append(f"Sentence {i}: {str(e)}")
            results["sentences"].append({
                "idx": i,
                "text": text,
                "error": str(e),
            })

    # Calculate statistics
    if durations:
        results["statistics"] = {
            "total_sentences": len(sentences),
            "successful": len(durations),
            "failed": len(sentences) - len(durations),
            "total_duration_sec": sum(durations),
            "avg_duration_sec": sum(durations) / len(durations),
            "avg_chars_per_sec": sum(lengths) / sum(durations) if sum(durations) > 0 else 0,
            "failure_rates": {
                key: count / max(len(durations), 1)
                for key, count in failure_counts.items()
            },
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TTS model")
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
        help="Path to NAR model checkpoint",
    )
    parser.add_argument(
        "--sentences",
        type=str,
        default=None,
        help="Path to file with test sentences (one per line)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval/output",
        help="Directory to save generated audio",
    )
    parser.add_argument(
        "--enable-gibberish",
        action="store_true",
        help="Enable Whisper-based gibberish detection (slower)",
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

    # Load sentences
    if args.sentences:
        with open(args.sentences) as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        sentences = DEFAULT_SENTENCES

    print(f"Evaluating on {len(sentences)} sentences")

    # Load models and create synthesizer
    codec = EnCodecBackend(bandwidth=6.0, device=device)
    tokenizer = CharTokenizer()

    # Load AR model
    ar_ckpt = load_checkpoint(args.ar_checkpoint, device=device)
    ar_config = ar_ckpt["config"]["model"]["ar"]
    ar_codec_cfg = ar_ckpt["config"].get("codec", {})
    _validate_checkpoint_config(ar_ckpt.get("config", {}), "AR")
    ar_contract = _resolve_contract(ar_ckpt, "ar", ar_config, ar_codec_cfg)

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

    # Load NAR model if provided
    nar_model = None
    if args.nar_checkpoint:
        nar_ckpt = load_checkpoint(args.nar_checkpoint, device=device)
        nar_config = nar_ckpt["config"]["model"]["nar"]
        nar_codec_cfg = nar_ckpt["config"].get("codec", ar_codec_cfg)
        _validate_checkpoint_config(nar_ckpt.get("config", {}), "NAR")
        nar_contract = _resolve_contract(nar_ckpt, "nar", nar_config, nar_codec_cfg)

        compat_errors = check_ar_nar_compatibility(
            ar_contract,
            nar_contract,
            ar_codec_cfg,
        )
        if compat_errors:
            lines = "\n".join(f"  - {err}" for err in compat_errors)
            raise ValueError(
                "Incompatible AR/NAR checkpoints detected:\n"
                f"{lines}\n"
                "Use checkpoints from the same experiment or retrain NAR with matching AR settings."
            )

        nar_model = NARTransformer(
            n_codebooks=nar_config.get("n_codebooks", 7),
            vocab_size=nar_config.get("vocab_size", 1024),
            d_model=nar_config.get("d_model", 512),
            n_heads=nar_config.get("n_heads", 8),
            n_layers=nar_config.get("n_layers", 8),
            d_ff=nar_config.get("d_ff", 2048),
            dropout=nar_config.get("dropout", 0.1),
            max_seq_len=nar_config.get("max_seq_len", ar_config.get("max_seq_len", 2048)),
            text_vocab_size=nar_config.get(
                "text_vocab_size", ar_config.get("text_vocab_size", tokenizer.vocab_size)
            ),
            max_text_len=nar_config.get("max_text_len", ar_config.get("max_text_len", 512)),
            text_padding_idx=nar_config.get("text_padding_idx", 0),
            n_text_layers=nar_config.get("n_text_layers", 4),
            use_speaker_conditioning=nar_config.get("use_speaker_conditioning", False),
            speaker_n_codebooks=nar_config.get(
                "speaker_n_codebooks",
                ar_ckpt.get("config", {}).get("codec", {}).get("n_codebooks", 8),
            ),
            speaker_pooling=nar_config.get("speaker_pooling", "mean"),
        )
        load_checkpoint(args.nar_checkpoint, model=nar_model, device=device)

    # Create synthesizer
    synthesizer = Synthesizer(
        ar_model=ar_model,
        nar_model=nar_model,
        codec=codec,
        tokenizer=tokenizer,
        device=device,
    )

    # Run evaluation
    output_dir = Path(args.output_dir)
    results = evaluate_model(
        synthesizer=synthesizer,
        sentences=sentences,
        reference_audio=args.reference,
        output_dir=output_dir,
        enable_gibberish=args.enable_gibberish,
    )

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n=== Evaluation Results ===")
    stats = results["statistics"]
    print(f"Successful: {stats.get('successful', 0)}/{stats.get('total_sentences', 0)}")
    print(f"Total duration: {stats.get('total_duration_sec', 0):.2f}s")
    print(f"Avg duration: {stats.get('avg_duration_sec', 0):.2f}s")
    print(f"Avg chars/sec: {stats.get('avg_chars_per_sec', 0):.1f}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results["errors"][:5]:
            print(f"  - {err}")

    print(f"\nResults saved to {results_path}")
    print(f"Audio files saved to {output_dir}")


if __name__ == "__main__":
    main()
