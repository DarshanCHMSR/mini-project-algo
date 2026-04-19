"""
Temporal models for ProBayes defect classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Conv1d with manual left-only padding to enforce causality."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual TCN block with two causal dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if in_channels != out_channels:
            self.residual_proj = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.conv_block(x)
        return F.relu(out + residual)


class TCNDefectClassifier(nn.Module):
    """TCN classifier for binary defect prediction with scalar feature fusion."""

    def __init__(
        self,
        in_channels: int,
        channel_list: list,
        kernel_size: int = 3,
        n_scalar_feats: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        blocks = []
        for i, out_ch in enumerate(channel_list):
            in_ch = in_channels if i == 0 else channel_list[i - 1]
            dilation = 2 ** i
            blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.tcn_blocks = nn.Sequential(*blocks)

        final_dim = channel_list[-1] + n_scalar_feats
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Requested derivation comment:
        # total receptive field = 1 + (kernel_size - 1) * sum(all dilation values across all blocks)
        self.receptive_field = 1 + (kernel_size - 1) * sum(2 ** i for i in range(len(channel_list)))

    def forward(self, x_ts: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        h = self.tcn_blocks(x_ts)
        h = h.mean(dim=2)
        combined = torch.cat([h, x_scalar], dim=1)
        out = self.classifier(combined)
        return out.squeeze(-1)


class FocalLoss(nn.Module):
    """Binary focal loss over logits for class-imbalanced learning."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class LSTMClassifier(nn.Module):
    """LSTM baseline with global average pooling and shared scalar head design."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_scalar_feats: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        final_dim = hidden_size + n_scalar_feats
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_ts: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        x_seq = x_ts.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_seq)
        h = lstm_out.mean(dim=1)
        combined = torch.cat([h, x_scalar], dim=1)
        out = self.classifier(combined)
        return out.squeeze(-1)
