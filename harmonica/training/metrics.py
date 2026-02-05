"""Training metrics tracking and logging."""

from collections import defaultdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time

import torch


class MetricsTracker:
    """Track and aggregate training metrics."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        use_tensorboard: bool = True,
    ):
        """Initialize metrics tracker.

        Args:
            log_dir: Directory for TensorBoard logs
            use_tensorboard: Enable TensorBoard logging
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.use_tensorboard = use_tensorboard

        # Running metrics for current logging interval
        self.running_metrics: Dict[str, List[float]] = defaultdict(list)

        # History of logged metrics
        self.history: Dict[str, List[tuple]] = defaultdict(list)  # (step, value)

        # Best metrics for checkpointing
        self.best_metrics: Dict[str, float] = {}

        # TensorBoard writer
        self.writer = None
        if use_tensorboard and log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                print("TensorBoard not available, skipping")

        # Timing
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def update(self, metrics: Dict[str, float]) -> None:
        """Update running metrics.

        Args:
            metrics: Dictionary of metric values
        """
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.running_metrics[name].append(value)

    def log(self, step: int) -> Dict[str, float]:
        """Log aggregated metrics and reset.

        Args:
            step: Current training step

        Returns:
            Dictionary of averaged metrics
        """
        # Aggregate running metrics
        aggregated = {}
        for name, values in self.running_metrics.items():
            if values:
                avg = sum(values) / len(values)
                aggregated[name] = avg
                self.history[name].append((step, avg))

        # Log to TensorBoard
        if self.writer:
            for name, value in aggregated.items():
                self.writer.add_scalar(name, value, step)

        # Compute throughput
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        steps_per_sec = len(self.running_metrics.get("loss", [1])) / max(elapsed, 1e-6)
        aggregated["steps_per_sec"] = steps_per_sec
        self.last_log_time = current_time

        # Reset running metrics
        self.running_metrics = defaultdict(list)

        return aggregated

    def is_best(self, metric_name: str, value: float, higher_is_better: bool = False) -> bool:
        """Check if value is best so far.

        Args:
            metric_name: Name of metric
            value: Current value
            higher_is_better: True if higher values are better

        Returns:
            True if this is the best value
        """
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = value
            return True

        best = self.best_metrics[metric_name]
        if higher_is_better:
            is_best = value > best
        else:
            is_best = value < best

        if is_best:
            self.best_metrics[metric_name] = value

        return is_best

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics.

        Returns:
            Dictionary with training summary
        """
        total_time = time.time() - self.start_time

        summary = {
            "total_time_sec": total_time,
            "total_time_str": format_time(total_time),
            "best_metrics": self.best_metrics.copy(),
        }

        # Add final values from history
        for name, hist in self.history.items():
            if hist:
                summary[f"final_{name}"] = hist[-1][1]

        return summary

    def save_history(self, path: str) -> None:
        """Save metrics history to JSON.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "history": {
                name: [(step, val) for step, val in hist]
                for name, hist in self.history.items()
            },
            "best_metrics": self.best_metrics,
            "summary": self.get_summary(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()

    def log_codebook_utilization(
        self,
        predicted_tokens: torch.Tensor,
        vocab_size: int,
        step: int,
        prefix: str = "train",
    ) -> None:
        """Log codebook utilization and token histogram."""
        if self.writer is None:
            return

        unique_tokens = torch.unique(predicted_tokens).numel()
        utilization = unique_tokens / max(vocab_size, 1)

        self.writer.add_scalar(f"{prefix}/codebook_utilization", utilization, step)

        token_counts = torch.bincount(
            predicted_tokens.flatten().cpu(), minlength=vocab_size
        )
        self.writer.add_histogram(
            f"{prefix}/token_distribution", token_counts, step
        )

        if utilization < 0.3:
            print(f"WARNING: Codebook collapse detected! Only {utilization:.1%} of vocab used")

    def log_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        step: int,
        prefix: str = "train",
    ) -> None:
        """Log attention entropy and diversity."""
        if self.writer is None:
            return

        # attention_weights: [B, H, L_q, L_k]
        entropy = -(attention_weights * torch.log(attention_weights + 1e-9)).sum(dim=-1)
        mean_entropy = entropy.mean().item()

        self.writer.add_scalar(f"{prefix}/attention_entropy", mean_entropy, step)

        head_entropy_std = entropy.mean(dim=[0, 2]).std().item()
        self.writer.add_scalar(f"{prefix}/attention_diversity", head_entropy_std, step)

        if head_entropy_std < 0.1:
            print("WARNING: Attention collapse detected! All heads doing similar things")

    def log_gradient_stats(
        self,
        model: torch.nn.Module,
        step: int,
        prefix: str = "train",
    ) -> None:
        """Log gradient norms."""
        if self.writer is None:
            return

        total_norm_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            param_norm = param.grad.data.norm(2).item()
            total_norm_sq += param_norm ** 2
            layer_name = name.split(".")[0]
            self.writer.add_scalar(
                f"{prefix}/grad_norm/{layer_name}",
                param_norm,
                step,
            )

        total_norm = total_norm_sq ** 0.5
        self.writer.add_scalar(f"{prefix}/grad_norm_total", total_norm, step)

        if total_norm > 10.0:
            print(f"WARNING: Large gradient norm detected: {total_norm:.2f}")
        elif total_norm < 0.001:
            print(f"WARNING: Vanishing gradients detected: {total_norm:.6f}")


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as string for logging.

    Args:
        metrics: Dictionary of metric values

    Returns:
        Formatted string
    """
    parts = []
    for name, value in sorted(metrics.items()):
        if isinstance(value, float):
            if abs(value) < 0.01 or abs(value) > 1000:
                parts.append(f"{name}: {value:.2e}")
            else:
                parts.append(f"{name}: {value:.4f}")
        else:
            parts.append(f"{name}: {value}")
    return " | ".join(parts)
