"""Optimizer and learning rate scheduler utilities."""

import math
from typing import Optional, Iterator

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """Create optimizer with weight decay handling.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas
        eps: Adam epsilon
        optimizer_type: "adamw" or "adam"

    Returns:
        Configured optimizer
    """
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for biases and layer norms
        if "bias" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optimizer_type == "adamw":
        return AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    elif optimizer_type == "adam":
        return Adam(param_groups, lr=lr, betas=betas, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 1000,
    max_steps: int = 100000,
    scheduler_type: str = "cosine_warmup",
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        scheduler_type: Type of scheduler
        min_lr_ratio: Minimum LR as ratio of initial LR

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine_warmup":
        return get_cosine_warmup_scheduler(
            optimizer, warmup_steps, max_steps, min_lr_ratio
        )
    elif scheduler_type == "linear_warmup":
        return get_linear_warmup_scheduler(
            optimizer, warmup_steps, max_steps
        )
    elif scheduler_type == "constant_warmup":
        return get_constant_warmup_scheduler(
            optimizer, warmup_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine annealing with linear warmup.

    LR starts at 0, linearly increases to max during warmup,
    then follows cosine decay to min_lr.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def get_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
) -> LambdaLR:
    """Linear warmup then linear decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            return max(0.0, 1 - (step - warmup_steps) / (max_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def get_constant_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> LambdaLR:
    """Linear warmup then constant LR."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


class GradScaler:
    """Gradient scaler for mixed precision training.

    Works with both CUDA AMP and MPS (where AMP is limited).
    """

    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """Initialize gradient scaler.

        Args:
            enabled: Enable scaling
            init_scale: Initial scale factor
            growth_factor: Scale growth factor
            backoff_factor: Scale backoff factor
            growth_interval: Steps between scale increases
        """
        self.enabled = enabled

        if enabled and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )
        else:
            self.scaler = None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer with unscaling."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Update scale factor."""
        if self.scaler is not None:
            self.scaler.update()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients for clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def get_scale(self) -> float:
        """Get current scale."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0

    def state_dict(self) -> dict:
        """Get state dict."""
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict."""
        if self.scaler is not None and state_dict:
            self.scaler.load_state_dict(state_dict)
