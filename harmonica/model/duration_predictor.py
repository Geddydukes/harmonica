"""Duration predictor for length control."""

import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    """Predict number of audio tokens per text token."""

    def __init__(
        self,
        text_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, text_encoding: torch.Tensor) -> torch.Tensor:
        """Predict per-token durations.

        Args:
            text_encoding: [B, T, D]

        Returns:
            durations: [B, T]
        """
        return self.model(text_encoding).squeeze(-1)

    def predict_total_length(self, text_encoding: torch.Tensor) -> torch.Tensor:
        """Predict total output length in tokens.

        Args:
            text_encoding: [B, T, D]

        Returns:
            total_length: [B]
        """
        durations = self.forward(text_encoding)
        return durations.sum(dim=1).round().long()
