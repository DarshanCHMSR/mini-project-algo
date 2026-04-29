"""
SE-GradCAM TCN — Squeeze-and-Excitation Temporal Convolutional Network
with Gradient-Weighted Phase-Aligned Saliency.

Architectural Contributions
============================
1. Squeeze-and-Excitation (SE) channel attention inside each TCNBlock:
   - SQUEEZE: global average pool over the time axis → one scalar per channel,
     capturing the global cycle context for that sensor.
   - EXCITE: two FC layers (ReLU bottleneck → Sigmoid output) produce
     per-channel weights in [0, 1].
   - SCALE: multiply each channel's full time-series by its learned weight,
     making the model explicitly learn which sensors are most diagnostic for
     defect prediction.

2. GradCAM temporal saliency (implemented in explainer.py):
   - Hooks on the final TCNBlock capture the output feature map (forward)
     and its gradient (backward).
   - Gradient-weighted combination of feature-map channels → ReLU → normalize
     produces a saliency curve over time showing when the model attended.
   - Causally valid: CausalConv1d uses left-only padding, so the feature map
     at timestep t encodes only information from timesteps ≤ t.
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
            padding=0,  # all padding is manual and left-only — never 'same'
        )
        self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))  # left-only pad preserves causality
        return self.conv(x)


class SqueezeExcitation(nn.Module):
    """
    Channel attention via global average pooling + two-layer FC gating.

    Squeeze: collapse time axis → (batch, channels) descriptor.
    Excite:  FC → ReLU → FC → Sigmoid → per-channel weights in [0, 1].
    Scale:   multiply each channel's time-series by its weight.
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        # Bottleneck size: floor(n_channels / reduction), but never less than 4
        bottleneck = max(n_channels // reduction, 4)
        self.excitation = nn.Sequential(
            nn.Linear(n_channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, n_channels),
            nn.Sigmoid(),  # weights must be in [0, 1] for interpretable scaling
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, channels, timesteps)
        squeeze = x.mean(dim=-1)                    # (batch, channels) — global cycle context
        weights = self.excitation(squeeze)           # (batch, channels) — learned channel importance
        scaled = x * weights.unsqueeze(-1)           # broadcast weights over time axis
        return scaled, weights                       # return weights for post-hoc inspection


class TCNBlock(nn.Module):
    """Residual TCN block with two causal dilated convolutions + SE attention."""

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

        # Squeeze-Excitation module for channel attention
        self.se = SqueezeExcitation(out_channels, reduction=4)
        # Cache for post-hoc SE weight inspection without affecting gradients during training.
        self.last_se_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        # Apply SE attention: channel-reweight the conv output
        out, se_weights = self.se(out)
        # Detach weights to prevent gradient accumulation in cache (avoids memory leaks)
        self.last_se_weights = se_weights.detach().cpu()
        # Residual connection + final ReLU activation
        return F.relu(out + self.residual_proj(x))


class TCNDefectClassifier(nn.Module):
    """SE-TCN classifier for binary defect prediction with scalar feature fusion."""

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
            dilation = 2 ** i  # exponential dilation: [1, 2, 4, ...]
            blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.tcn_blocks = nn.Sequential(*blocks)

        final_dim = channel_list[-1] + n_scalar_feats
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Receptive field = 1 + (kernel_size - 1) * sum(all dilations across all blocks)
        # For kernel_size=3, dilations=[1,2,4]: RF = 1 + (3-1)*(1+2+4) = 15 timesteps
        self.receptive_field = 1 + (kernel_size - 1) * sum(2 ** i for i in range(len(channel_list)))

    def forward(self, x_ts: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        h = self.tcn_blocks(x_ts)
        h = h.mean(dim=2)                               # global average pool over time
        combined = torch.cat([h, x_scalar], dim=1)      # fuse scalar features
        out = self.classifier(combined)
        # Return single logit per sample — NO sigmoid applied here
        return out.squeeze(-1)

    def get_se_weights(self) -> list:
        """
        Return cached SE weights from the most recent forward pass.
        Returns a list of tensors, one per TCNBlock that executed.
        Returns empty list if no forward pass has run yet.
        """
        weights = []
        for block in self.tcn_blocks:
            if isinstance(block, TCNBlock) and block.last_se_weights is not None:
                weights.append(block.last_se_weights)
        return weights


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


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric focal loss with separate gamma values for positive/negative samples.

    gamma_pos > gamma_neg because missing a defect (false negative) is more costly
    than falsely flagging a good part (false positive) in a manufacturing context.
    Higher gamma on positives down-weights easy positive examples more aggressively,
    forcing the model to focus harder on hard-to-detect defects.
    """

    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 4.0, alpha: float = 0.75, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg   # focal exponent for negative samples (OK parts)
        self.gamma_pos = gamma_pos   # focal exponent for positive samples (defective parts)
        self.alpha = alpha           # weighting factor for positive class
        self.clip = clip             # probability clipping for numerical stability

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # Convert logits to probabilities
        prob = torch.sigmoid(logits)
        # Clamp to avoid log(0) numerical instability
        prob = torch.clamp(prob, self.clip, 1.0 - self.clip)

        # Positive branch: -alpha * (1 - p)^gamma_pos * log(p)
        # Higher gamma_pos forces model to focus on hard defects
        loss_pos = -self.alpha * (1.0 - prob) ** self.gamma_pos * torch.log(prob)
        # Negative branch: -(1 - alpha) * p^gamma_neg * log(1 - p)
        # Standard focal weighting for OK parts
        loss_neg = -(1.0 - self.alpha) * prob ** self.gamma_neg * torch.log(1.0 - prob)

        # Combine: targets weight positive loss, (1 - targets) weight negative loss
        loss = targets * loss_pos + (1.0 - targets) * loss_neg
        return loss.mean()


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric focal loss with separate gamma values for positive/negative samples.

    gamma_pos > gamma_neg because missing a defect (false negative) is more costly
    than falsely flagging a good part (false positive) in a manufacturing context.
    Higher gamma on positives down-weights easy positive examples more aggressively,
    forcing the model to focus on hard-to-detect defects.
    """

    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 4.0, alpha: float = 0.75, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.alpha = alpha
        self.clip = clip  # probability clipping prevents log(0) instability

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        prob = torch.sigmoid(logits)
        prob = torch.clamp(prob, self.clip, 1.0 - self.clip)   # numerical stability

        # Positive branch: penalise missed defects more heavily via gamma_pos
        loss_pos = -self.alpha * (1.0 - prob) ** self.gamma_pos * torch.log(prob)
        # Negative branch: standard focal weighting for OK parts
        loss_neg = -(1.0 - self.alpha) * prob ** self.gamma_neg * torch.log(1.0 - prob)

        loss = targets * loss_pos + (1.0 - targets) * loss_neg
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
