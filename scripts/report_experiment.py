#!/usr/bin/env python3
"""Generate a concise markdown report for a training run."""

import argparse
import json
from pathlib import Path
from datetime import datetime


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _best_from_history(history: dict, metric: str, lower_is_better: bool = True):
    series = history.get(metric, [])
    if not series:
        return None, None
    best = min(series, key=lambda x: x[1]) if lower_is_better else max(series, key=lambda x: x[1])
    return best[0], best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment markdown report")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to experiment log dir")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to checkpoint dir (defaults to sibling of log dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path (default: <log-dir>/report.md)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else log_dir.parent.parent / "checkpoints" / log_dir.name
    output_path = Path(args.output) if args.output else log_dir / "report.md"

    metrics_path = log_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    metrics = _load_json(metrics_path)
    history = metrics.get("history", {})
    summary = metrics.get("summary", {})
    best_metrics = metrics.get("best_metrics", {})

    best_ckpt_json = ckpt_dir / "checkpoint_best.json"
    best_ckpt = _load_json(best_ckpt_json) if best_ckpt_json.exists() else {}

    best_val_step, best_val = _best_from_history(history, "val_loss", lower_is_better=True)
    best_ema_step, best_ema = _best_from_history(history, "val_ema_loss", lower_is_better=True)

    lines = [
        "# Experiment Report",
        "",
        f"- Generated: {datetime.now().isoformat()}",
        f"- Log dir: `{log_dir}`",
        f"- Checkpoint dir: `{ckpt_dir}`",
        "",
        "## Summary",
        f"- Total time: `{summary.get('total_time_str', 'n/a')}`",
        f"- Final train loss: `{summary.get('final_loss', 'n/a')}`",
        f"- Final val loss: `{summary.get('final_val_loss', 'n/a')}`",
        f"- Final val EMA loss: `{summary.get('final_val_ema_loss', 'n/a')}`",
        "",
        "## Best Metrics",
    ]

    if best_metrics:
        for key in sorted(best_metrics):
            lines.append(f"- {key}: `{best_metrics[key]}`")
    else:
        lines.append("- None recorded")

    lines.extend(
        [
            "",
            "## Best History Points",
            f"- Best `val_loss`: step `{best_val_step}` value `{best_val}`",
            f"- Best `val_ema_loss`: step `{best_ema_step}` value `{best_ema}`",
            "",
            "## Best Checkpoint",
            f"- Path: `{ckpt_dir / 'checkpoint_best.pt'}`",
            f"- Step: `{best_ckpt.get('step', 'n/a')}`",
            f"- Optimizer step: `{best_ckpt.get('optimizer_step', 'n/a')}`",
            f"- Timestamp: `{best_ckpt.get('timestamp', 'n/a')}`",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
