"""Main training loop for Harmonica."""

from pathlib import Path
from typing import Any, Dict, Optional, Callable
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .optimizer import create_optimizer, create_scheduler, GradScaler
from .checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint, cleanup_checkpoints
from .metrics import MetricsTracker, format_metrics
from ..model import ARTransformer
try:
    from eval.detect_failures import FailureModeDetector
except Exception:
    FailureModeDetector = None
from ..data.collate import HarmonicaBatch
from ..utils.device import get_device, supports_mixed_precision
from ..utils.seed import get_random_state


class Trainer:
    """Training loop for AR and NAR models."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train (ARTransformer or NARTransformer)
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device(config.get("device", {}).get("prefer", "auto"))

        # Move model to device
        self.model.to(self.device)

        # Training config
        train_cfg = config.get("training", {})
        self.grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.label_smoothing = train_cfg.get("label_smoothing", 0.0)
        self.use_scheduled_sampling = train_cfg.get("scheduled_sampling", False)
        self.teacher_forcing_schedule = train_cfg.get("teacher_forcing_schedule", "linear")
        self.teacher_forcing_start = train_cfg.get("teacher_forcing_start", 1.0)
        self.teacher_forcing_end = train_cfg.get("teacher_forcing_end", 0.3)
        self.duration_loss_weight = train_cfg.get("duration_loss_weight", 0.0)
        self.codebook_entropy_weight = train_cfg.get("codebook_entropy_weight", 0.0)
        self.token_noise_prob = train_cfg.get("token_noise_prob", 0.0)

        # Step semantics: schedules and intervals are tracked in optimizer-update units.
        self.max_update_steps = self._resolve_update_steps(
            train_cfg, "max_update_steps", "max_steps", default_legacy=100000, allow_zero=False
        )
        self.warmup_update_steps = self._resolve_update_steps(
            train_cfg, "warmup_update_steps", "warmup_steps", default_legacy=1000, allow_zero=True
        )
        self.log_every_updates = self._resolve_update_steps(
            train_cfg, "log_every_updates", "log_every", default_legacy=100, allow_zero=False
        )
        self.eval_every_updates = self._resolve_update_steps(
            train_cfg, "eval_every_updates", "eval_every", default_legacy=1000, allow_zero=False
        )
        self.checkpoint_every_updates = self._resolve_update_steps(
            train_cfg,
            "checkpoint_every_updates",
            "checkpoint_every",
            default_legacy=5000,
            allow_zero=False,
        )
        self.warn_after_updates = self._resolve_update_steps(
            train_cfg, "warn_after_updates", "warn_after_step", default_legacy=2000, allow_zero=True
        )
        self.codebook_entropy_warmup_updates = self._resolve_update_steps(
            train_cfg,
            "codebook_entropy_warmup_updates",
            "codebook_entropy_warmup_steps",
            default_legacy=0,
            allow_zero=True,
        )
        self.token_noise_warmup_updates = self._resolve_update_steps(
            train_cfg,
            "token_noise_warmup_updates",
            "token_noise_warmup_steps",
            default_legacy=0,
            allow_zero=True,
        )

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(
            model,
            lr=train_cfg.get("lr", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=self.warmup_update_steps,
            max_steps=self.max_update_steps,
            scheduler_type=train_cfg.get("scheduler_type", "cosine_warmup"),
            min_lr_ratio=train_cfg.get("min_lr_ratio", 0.1),
        )

        # Mixed precision
        use_mixed = train_cfg.get("mixed_precision", True) and supports_mixed_precision(self.device)
        self.scaler = GradScaler(enabled=use_mixed and self.device.type == "cuda")
        self.use_autocast = use_mixed

        # Experiment config
        exp_cfg = config.get("experiment", {})
        self.checkpoint_dir = Path(exp_cfg.get("checkpoint_dir", "./experiments/checkpoints"))
        self.log_dir = Path(exp_cfg.get("log_dir", "./experiments/logs"))

        # Metrics
        self.metrics = MetricsTracker(str(self.log_dir), use_tensorboard=True)

        # Training state
        self.micro_step = 0
        self.optimizer_step = 0
        self.global_step = 0  # legacy alias for micro_step
        self.best_val_loss = float("inf")
        self._last_pred_tokens = None
        self._last_audio_lengths = None

        # Optional failure detection for eval pipelines
        self.failure_detector = FailureModeDetector() if FailureModeDetector else None

    def _resolve_update_steps(
        self,
        train_cfg: Dict[str, Any],
        update_key: str,
        legacy_key: str,
        default_legacy: int,
        allow_zero: bool = False,
    ) -> int:
        """Resolve update-based values with backward-compatible micro-step conversion."""
        if update_key in train_cfg and train_cfg.get(update_key) is not None:
            value = int(train_cfg.get(update_key))
        else:
            legacy_value = int(train_cfg.get(legacy_key, default_legacy))
            if allow_zero and legacy_value <= 0:
                value = 0
            else:
                value = int(math.ceil(legacy_value / max(1, self.grad_accum_steps)))
            print(
                f"Info: converted `{legacy_key}={legacy_value}` micro-steps to "
                f"`{update_key}={value}` updates (grad_accum_steps={self.grad_accum_steps})."
            )

        if allow_zero:
            return max(0, value)
        return max(1, value)

    def train(self, resume: bool = True) -> Dict[str, Any]:
        """Run training loop.

        Args:
            resume: Resume from latest checkpoint if available

        Returns:
            Training summary
        """
        # Resume from checkpoint
        if resume:
            latest = get_latest_checkpoint(str(self.checkpoint_dir))
            if latest:
                print(f"Resuming from {latest}")
                ckpt_info = load_checkpoint(
                    str(latest),
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    device=self.device,
                    sampler=self.train_loader.sampler if self.train_loader is not None else None,
                )
                self.micro_step = int(ckpt_info.get("micro_step", ckpt_info.get("step", 0)))
                self.optimizer_step = int(
                    ckpt_info.get("optimizer_step", max(getattr(self.scheduler, "last_epoch", 0), 0))
                )
                aligned_micro = self.optimizer_step * self.grad_accum_steps
                if self.micro_step != aligned_micro:
                    print(
                        f"Warning: checkpoint micro_step ({self.micro_step}) does not match "
                        f"optimizer_step*grad_accum ({aligned_micro}). Aligning to {aligned_micro}."
                    )
                    self.micro_step = aligned_micro
                self.global_step = self.micro_step
                print(
                    f"Resumed at micro_step={self.micro_step}, optimizer_step={self.optimizer_step}"
                )

        # Training loop
        self.model.train()
        train_iter = iter(self.train_loader)
        accumulated_loss = 0.0
        micro_since_update = 0

        progress = tqdm(
            total=self.max_update_steps,
            initial=self.optimizer_step,
            desc="Training (updates)",
        )

        while self.optimizer_step < self.max_update_steps:
            next_is_update = (micro_since_update + 1) >= self.grad_accum_steps
            next_update_step = self.optimizer_step + 1
            log_attn = (
                self.metrics.writer is not None
                and next_is_update
                and next_update_step % self.log_every_updates == 0
            )
            self._set_attention_logging(log_attn)

            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Move batch to device
            batch = batch.to(self.device)

            # Forward pass with optional mixed precision
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_autocast,
            ):
                loss, metrics = self._compute_loss(batch)
                loss = loss / self.grad_accum_steps

            # Backward pass
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            micro_since_update += 1
            self.micro_step += 1
            self.global_step = self.micro_step

            # Optimizer step (after gradient accumulation)
            if micro_since_update >= self.grad_accum_steps:
                # Unscale for gradient clipping
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    metrics["grad_norm"] = grad_norm.item()

                log_step = next_update_step
                if self.metrics.writer and log_step % self.log_every_updates == 0:
                    self.metrics.log_gradient_stats(self.model, step=log_step, prefix="train")

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Scheduler step
                self.scheduler.step()
                self.optimizer_step += 1
                progress.update(1)
                micro_since_update = 0

                # Record loss (already averaged over grad_accum_steps)
                metrics["loss"] = accumulated_loss
                metrics["lr"] = self.scheduler.get_last_lr()[0]
                metrics["micro_step"] = float(self.micro_step)
                metrics["optimizer_step"] = float(self.optimizer_step)
                self.metrics.update(metrics)
                accumulated_loss = 0.0

                # Diagnostics logging
                if self.metrics.writer and log_step % self.log_every_updates == 0:
                    if self._last_pred_tokens is not None:
                        self.metrics.log_codebook_utilization(
                            self._last_pred_tokens,
                            vocab_size=self.model.vocab_size if hasattr(self.model, "vocab_size") else 1024,
                            step=log_step,
                            prefix="train",
                            pad_token_id=getattr(self.model, "pad_token_id", None),
                            audio_lengths=self._last_audio_lengths,
                            warn_after_step=self.warn_after_updates,
                        )
                    attn = None
                    if hasattr(self.model, "layers") and self.model.layers:
                        layer = self.model.layers[-1]
                        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "last_attn_weights"):
                            attn = layer.self_attn.last_attn_weights
                    if attn is not None:
                        self.metrics.log_attention_entropy(
                            attn,
                            step=log_step,
                            prefix="train",
                            warn_after_step=self.warn_after_updates,
                        )

                # Logging
                if self.optimizer_step % self.log_every_updates == 0:
                    logged = self.metrics.log(self.optimizer_step)
                    progress.set_postfix(
                        loss=f"{logged.get('loss', 0):.4f}",
                        lr=f"{logged.get('lr', 0):.2e}",
                    )

                # Evaluation
                if self.val_loader and self.optimizer_step % self.eval_every_updates == 0:
                    val_metrics = self.evaluate()
                    self.metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                    if val_metrics:
                        val_loss = val_metrics.get("loss")
                        val_ppl = val_metrics.get("perplexity")
                        msg = f"Eval @ update {self.optimizer_step}: val_loss={val_loss:.4f}"
                        if val_ppl is not None:
                            msg += f", val_ppl={val_ppl:.2f}"
                        print(msg)

                    # Check for best model
                    if val_metrics.get("loss", float("inf")) < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self._save_checkpoint("checkpoint_best.pt")

                # Checkpointing
                if self.optimizer_step % self.checkpoint_every_updates == 0:
                    self._save_checkpoint(f"checkpoint_{self.micro_step}.pt")
                    cleanup_checkpoints(str(self.checkpoint_dir), keep_last=5)

        # Final checkpoint
        self._save_checkpoint(f"checkpoint_{self.micro_step}.pt")

        # Save metrics history
        self.metrics.save_history(str(self.log_dir / "metrics.json"))
        self.metrics.close()

        return self.metrics.get_summary()

    def _compute_loss(self, batch: HarmonicaBatch) -> tuple:
        """Compute loss for a batch.

        Override this method for different model types.

        Args:
            batch: Training batch

        Returns:
            Tuple of (loss, metrics dict)
        """
        # Get first codebook tokens for AR model
        audio_tokens = batch.codec_tokens[:, 0, :]  # [B, L]

        # Prompt tokens if available (first codebook only)
        prompt_tokens = None
        prompt_lengths = None
        if batch.prompt_tokens is not None:
            prompt_tokens = batch.prompt_tokens[:, 0, :]
            prompt_lengths = batch.prompt_lengths

        if isinstance(self.model, ARTransformer):
            if self.use_scheduled_sampling:
                loss, metrics = self._ar_scheduled_sampling_step(
                    audio_tokens=audio_tokens,
                    text_tokens=batch.text_tokens,
                    text_lengths=batch.text_lengths,
                    prompt_tokens=prompt_tokens,
                    audio_lengths=batch.audio_lengths,
                )
            else:
                loss, metrics = self._ar_teacher_forcing_step(
                    audio_tokens=audio_tokens,
                    text_tokens=batch.text_tokens,
                    text_lengths=batch.text_lengths,
                    prompt_tokens=prompt_tokens,
                    audio_lengths=batch.audio_lengths,
                )
            return loss, metrics

        # Default path for other models
        loss, metrics = self.model.compute_loss(
            audio_tokens=audio_tokens,
            text_tokens=batch.text_tokens,
            text_lengths=batch.text_lengths,
            prompt_tokens=prompt_tokens,
            prompt_lengths=prompt_lengths,
            audio_lengths=batch.audio_lengths,
            label_smoothing=self.label_smoothing,
        )

        return loss, metrics

    def get_teacher_forcing_ratio(self, step: int, max_steps: int) -> float:
        """Compute teacher forcing ratio for current step."""
        if not self.use_scheduled_sampling:
            return 1.0

        progress = min(step / max_steps, 1.0)

        if self.teacher_forcing_schedule == "linear":
            ratio = self.teacher_forcing_start - progress * (
                self.teacher_forcing_start - self.teacher_forcing_end
            )
        elif self.teacher_forcing_schedule == "exponential":
            decay_rate = -np.log(self.teacher_forcing_end / self.teacher_forcing_start)
            ratio = self.teacher_forcing_start * np.exp(-decay_rate * progress)
        else:
            ratio = self.teacher_forcing_start

        return float(ratio)

    def _ar_teacher_forcing_step(
        self,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor],
        prompt_tokens: Optional[torch.Tensor],
        audio_lengths: Optional[torch.Tensor],
    ) -> tuple:
        """Standard teacher forcing step with logging hooks."""
        if self.model.stop_token_id is not None:
            stop = torch.full(
                (audio_tokens.shape[0], 1),
                self.model.stop_token_id,
                device=audio_tokens.device,
                dtype=audio_tokens.dtype,
            )
            audio_tokens = torch.cat([audio_tokens, stop], dim=1)
            if audio_lengths is not None:
                audio_lengths = audio_lengths + 1

        noisy_tokens = self._maybe_apply_token_noise(audio_tokens, audio_lengths)
        logits = self.model.forward(
            audio_tokens=noisy_tokens,
            text_tokens=text_tokens,
            text_lengths=text_lengths,
            prompt_tokens=prompt_tokens,
        )

        B, L, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = audio_tokens.reshape(-1)

        loss = torch.nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            label_smoothing=self.label_smoothing,
            reduction="none",
        ).reshape(B, L)

        if audio_lengths is not None:
            mask = torch.arange(L, device=audio_tokens.device).expand(B, L) < audio_lengths.unsqueeze(1)
            loss = (loss * mask.float()).sum() / mask.float().sum()
        else:
            loss = loss.mean()

        entropy_loss = self._maybe_entropy_regularize(logits, audio_lengths)
        if entropy_loss is not None:
            loss = loss + entropy_loss

        duration_loss = None
        if (
            hasattr(self.model, "duration_predictor")
            and self.duration_loss_weight > 0
            and audio_lengths is not None
            and text_lengths is not None
        ):
            text_emb, text_mask = self.model.encode_text(text_tokens, text_lengths)
            target_per_token = audio_lengths.float() / text_lengths.float()
            target = target_per_token.unsqueeze(1).expand_as(text_emb[..., 0])
            pred = self.model.duration_predictor(text_emb)
            mask = (~text_mask).float() if text_mask is not None else torch.ones_like(pred)
            duration_loss = (torch.abs(pred - target) * mask).sum() / mask.sum()
            loss = loss + self.duration_loss_weight * duration_loss

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            self._last_pred_tokens = preds.detach()
            self._last_audio_lengths = audio_lengths.detach() if audio_lengths is not None else None
            accuracy = (preds == audio_tokens).float().mean()
            perplexity = torch.exp(loss.detach())

        metrics = {
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item(),
        }
        if entropy_loss is not None:
            metrics["codebook_entropy_loss"] = entropy_loss.item()
        if duration_loss is not None:
            metrics["duration_loss"] = duration_loss.item()
        return loss, metrics

    def _maybe_apply_token_noise(
        self,
        audio_tokens: torch.Tensor,
        audio_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply input token noise during teacher forcing to reduce collapse."""
        prob = self._token_noise_prob_for_step(self.optimizer_step)
        if prob <= 0:
            return audio_tokens

        B, L = audio_tokens.shape
        device = audio_tokens.device

        if audio_lengths is not None:
            valid = torch.arange(L, device=device).expand(B, L) < audio_lengths.unsqueeze(1)
        else:
            valid = torch.ones((B, L), device=device, dtype=torch.bool)

        noise_mask = (torch.rand(B, L, device=device) < prob) & valid
        if not noise_mask.any():
            return audio_tokens

        random_tokens = torch.randint(
            low=0,
            high=self.model.vocab_size,
            size=(B, L),
            device=device,
            dtype=audio_tokens.dtype,
        )
        return torch.where(noise_mask, random_tokens, audio_tokens)

    def _token_noise_prob_for_step(self, step: int) -> float:
        """Ramp token noise down after warmup."""
        if self.token_noise_prob <= 0:
            return 0.0
        if self.token_noise_warmup_updates <= 0:
            return float(self.token_noise_prob)
        if step >= self.token_noise_warmup_updates:
            return 0.0
        remaining = 1.0 - (step / float(self.token_noise_warmup_updates))
        return float(self.token_noise_prob * max(0.0, remaining))

    def _ar_scheduled_sampling_step(
        self,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor],
        prompt_tokens: Optional[torch.Tensor],
        audio_lengths: Optional[torch.Tensor],
    ) -> tuple:
        """AR training with scheduled sampling."""
        teacher_forcing_ratio = self.get_teacher_forcing_ratio(
            self.optimizer_step, self.max_update_steps
        )

        text_emb, text_mask = self.model.encode_text(text_tokens, text_lengths)

        if self.model.stop_token_id is not None:
            stop = torch.full(
                (audio_tokens.shape[0], 1),
                self.model.stop_token_id,
                device=audio_tokens.device,
                dtype=audio_tokens.dtype,
            )
            audio_tokens = torch.cat([audio_tokens, stop], dim=1)
            if audio_lengths is not None:
                audio_lengths = audio_lengths + 1

        B, L = audio_tokens.shape
        device = audio_tokens.device

        all_logits = torch.empty(B, L, self.model.vocab_size, device=device)
        all_targets = audio_tokens

        generated_tokens = torch.empty(B, L, device=device, dtype=audio_tokens.dtype)

        for t in range(L):
            prev_tokens = generated_tokens[:, :t].clone() if t > 0 else None
            logits = self.model.forward_step(
                prev_tokens=prev_tokens,
                text_emb=text_emb,
                text_mask=text_mask,
                prompt_tokens=prompt_tokens,
            )
            all_logits[:, t, :] = logits

            use_gt = torch.rand(B, device=device) < teacher_forcing_ratio
            predicted = logits.argmax(dim=-1)
            next_token = torch.where(use_gt, audio_tokens[:, t], predicted)

            generated_tokens[:, t] = next_token

        loss = torch.nn.functional.cross_entropy(
            all_logits.reshape(-1, self.model.vocab_size),
            all_targets.reshape(-1),
            label_smoothing=self.label_smoothing,
            reduction="none",
        ).reshape(B, L)

        if audio_lengths is not None:
            mask = torch.arange(L, device=device).expand(B, L) < audio_lengths.unsqueeze(1)
            loss = (loss * mask.float()).sum() / mask.float().sum()
        else:
            loss = loss.mean()

        entropy_loss = self._maybe_entropy_regularize(all_logits, audio_lengths)
        if entropy_loss is not None:
            loss = loss + entropy_loss

        duration_loss = None
        if (
            hasattr(self.model, "duration_predictor")
            and self.duration_loss_weight > 0
            and audio_lengths is not None
            and text_lengths is not None
        ):
            target_per_token = audio_lengths.float() / text_lengths.float()
            target = target_per_token.unsqueeze(1).expand_as(text_emb[..., 0])
            pred = self.model.duration_predictor(text_emb)
            mask = (~text_mask).float() if text_mask is not None else torch.ones_like(pred)
            duration_loss = (torch.abs(pred - target) * mask).sum() / mask.sum()
            loss = loss + self.duration_loss_weight * duration_loss

        with torch.no_grad():
            preds = all_logits.argmax(dim=-1)
            self._last_pred_tokens = preds.detach()
            accuracy = (preds == all_targets).float().mean()
            perplexity = torch.exp(loss.detach())

        metrics = {
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item(),
            "teacher_forcing_ratio": teacher_forcing_ratio,
        }
        if entropy_loss is not None:
            metrics["codebook_entropy_loss"] = entropy_loss.item()
        if duration_loss is not None:
            metrics["duration_loss"] = duration_loss.item()

        return loss, metrics

    def _maybe_entropy_regularize(
        self,
        logits: torch.Tensor,
        audio_lengths: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Encourage higher entropy over codebook logits early in training."""
        weight = self._codebook_entropy_weight_for_step(self.optimizer_step)
        if weight <= 0:
            return None

        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B, L]
        if audio_lengths is not None:
            B, L = entropy.shape
            mask = torch.arange(L, device=entropy.device).expand(B, L) < audio_lengths.unsqueeze(1)
            entropy = (entropy * mask.float()).sum() / mask.float().sum()
        else:
            entropy = entropy.mean()

        # Negative entropy to maximize entropy
        return weight * (-entropy)

    def _codebook_entropy_weight_for_step(self, step: int) -> float:
        """Ramp entropy regularization down after warmup."""
        if self.codebook_entropy_weight <= 0:
            return 0.0
        if self.codebook_entropy_warmup_updates <= 0:
            return float(self.codebook_entropy_weight)
        if step >= self.codebook_entropy_warmup_updates:
            return 0.0
        # Linear decay to 0 over warmup window
        remaining = 1.0 - (step / float(self.codebook_entropy_warmup_updates))
        return float(self.codebook_entropy_weight * max(0.0, remaining))

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set.

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_autocast,
            ):
                loss, metrics = self._compute_loss(batch)

            total_loss += loss.item()
            total_acc += metrics.get("accuracy", 0)
            n_batches += 1

        self.model.train()

        return {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": total_acc / max(n_batches, 1),
            "perplexity": torch.exp(torch.tensor(total_loss / max(n_batches, 1))).item(),
        }

    def _save_checkpoint(self, filename: str) -> None:
        """Save checkpoint.

        Args:
            filename: Checkpoint filename
        """
        training_time = time.time() - self.metrics.start_time
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            optimizer_step=self.optimizer_step,
            micro_step=self.micro_step,
            config=self.config,
            path=str(self.checkpoint_dir / filename),
            scaler=self.scaler,
            metrics=self.metrics.best_metrics,
            random_state=get_random_state(),
            sampler=self.train_loader.sampler if self.train_loader is not None else None,
            training_time=training_time,
        )

    def _set_attention_logging(self, enabled: bool) -> None:
        """Toggle storing attention weights for diagnostics."""
        if not hasattr(self.model, "layers"):
            return
        for layer in self.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "store_attn"):
                layer.self_attn.store_attn = enabled


class NARTrainer(Trainer):
    """Trainer specialized for NAR model."""

    def _compute_loss(self, batch: HarmonicaBatch) -> tuple:
        """Compute loss for NAR model.

        Args:
            batch: Training batch

        Returns:
            Tuple of (loss, metrics dict)
        """
        return self.nar_training_step(batch)

    def nar_training_step(self, batch: HarmonicaBatch) -> tuple:
        """NAR training with ground truth codebook 1."""
        # CRITICAL: use ground-truth codebook 1 during training
        ar_tokens = batch.codec_tokens[:, 0, :]  # [B, L]
        target_tokens = batch.codec_tokens[:, 1:, :]  # [B, K, L]

        text_emb, text_mask = self.model.text_encoder(
            batch.text_tokens, batch.text_lengths
        )

        loss, metrics = self.model.compute_loss(
            ar_tokens=ar_tokens,
            target_tokens=target_tokens,
            text_emb=text_emb,
            text_mask=text_mask,
            audio_lengths=batch.audio_lengths,
            label_smoothing=self.label_smoothing,
        )

        return loss, metrics
