"""
GradCAM-based temporal saliency explainer for SE-GradCAM TCN.

GradCAMExplainer registers hooks on the final TCNBlock to capture:
  - forward: the output feature map (batch, channels, timesteps)
  - backward: the gradient of the scalar output w.r.t. that feature map

The gradient-weighted channel combination + ReLU produces a saliency curve
that is causally valid because CausalConv1d uses left-only padding.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class GradCAMExplainer:
    """Gradient-weighted Class Activation Map over the time axis of the final TCNBlock."""

    def __init__(self, model):
        self.model = model
        self.feature_map = None   # captured during forward pass
        self.gradient = None      # captured during backward pass
        self._forward_handle = None
        self._backward_handle = None

    def _register_hooks(self) -> None:
        last_block = self.model.tcn_blocks[-1]  # hook only the final block for highest-level features

        def forward_hook(module, input, output):
            # output is the block's return value: (batch, channels, timesteps)
            self.feature_map = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: gradient of loss w.r.t. this block's output
            self.gradient = grad_output[0].detach()

        self._forward_handle = last_block.register_forward_hook(forward_hook)
        self._backward_handle = last_block.register_full_backward_hook(backward_hook)

    def _remove_hooks(self) -> None:
        """Always call after explain() to prevent dangling hook memory leaks."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def explain(self, x_ts: torch.Tensor, x_scalar: torch.Tensor) -> np.ndarray:
        """
        Compute GradCAM saliency for a single sample.

        Args:
            x_ts:     (1, 8, T_max) — time-series input, must have requires_grad=True
            x_scalar: (1, 5)        — scalar features

        Returns:
            saliency: (T_max,) numpy array normalised to [0, 1]
        """
        self._register_hooks()
        self.model.eval()          # eval mode — no dropout, no batch norm updates
        self.model.zero_grad()

        logit = self.model(x_ts, x_scalar)   # forward pass triggers forward_hook
        logit.backward()                      # backward pass triggers backward_hook

        self._remove_hooks()  # clean up immediately — do not retain graph beyond this point

        # Average gradient over channels → importance weight per timestep
        # gradient shape: (batch, channels, timesteps) → weights: (batch, timesteps)
        weights = self.gradient.mean(dim=1)

        # Weighted sum of feature map channels: (batch, timesteps)
        cam = (weights.unsqueeze(1) * self.feature_map).sum(dim=1)

        cam = F.relu(cam)  # retain only positively contributing timesteps

        # Normalise per sample to [0, 1] so plots are comparable across cycles
        cam_min = cam.min(dim=-1, keepdim=True).values
        cam_max = cam.max(dim=-1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.squeeze(0).numpy()  # (T_max,)


def plot_saliency_overlay(
    raw_pressure: np.ndarray,
    saliency: np.ndarray,
    trig_inj: np.ndarray,
    trig_hld: np.ndarray,
    trig_cool: np.ndarray,
    lbl_nok: int,
    cycle_id: str,
    se_weights_per_block: Optional[List[np.ndarray]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Produce a multi-panel saliency overlay figure for one injection cycle.

    Panel 1: Raw injection pressure with phase background shading.
    Panel 2: GradCAM temporal saliency curve.
    Panel 3 (optional): SE channel weights per TCN block as grouped bar chart.
    """
    n_rows = 3 if se_weights_per_block is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=True)

    # Time axis in seconds: DXP signals sampled at 200 Hz → 0.005 s per step
    t = np.arange(len(raw_pressure)) * 0.005

    label_text = "DEFECTIVE (LBL_NOK=1)" if lbl_nok == 1 else "OK (LBL_NOK=0)"
    label_color = "red" if lbl_nok == 1 else "green"
    fig.suptitle(f"Cycle {cycle_id} — {label_text}", color=label_color, fontsize=13, fontweight="bold")

    # Phase colour constants used across all panels
    PHASE_COLORS = {
        "inj":  ("#3498DB", 0.12, "Injection"),
        "hld":  ("#E67E22", 0.12, "Holding"),
        "cool": ("#27AE60", 0.12, "Cooling"),
    }

    def _shade_phases(ax):
        """Apply phase background shading using axes-fraction y coordinates."""
        xform = ax.get_xaxis_transform()
        ax.fill_between(t, 0, 1, where=trig_inj.astype(bool),
                        color=PHASE_COLORS["inj"][0], alpha=PHASE_COLORS["inj"][1],
                        transform=xform, label="_nolegend_")
        ax.fill_between(t, 0, 1, where=trig_hld.astype(bool),
                        color=PHASE_COLORS["hld"][0], alpha=PHASE_COLORS["hld"][1],
                        transform=xform, label="_nolegend_")
        ax.fill_between(t, 0, 1, where=trig_cool.astype(bool),
                        color=PHASE_COLORS["cool"][0], alpha=PHASE_COLORS["cool"][1],
                        transform=xform, label="_nolegend_")

    # ── Panel 1: Raw pressure ──────────────────────────────────────────────
    ax1 = axes[0]
    _shade_phases(ax1)
    pressure_line, = ax1.plot(t, raw_pressure, color="#2C3E50", linewidth=1.2, label="Pressure")
    ax1.set_ylabel("Pressure (bar)")
    ax1.grid(alpha=0.3)

    # Legend: phase patches + pressure line
    legend_handles = [
        mpatches.Patch(color=PHASE_COLORS["inj"][0],  alpha=0.5, label="Injection"),
        mpatches.Patch(color=PHASE_COLORS["hld"][0],  alpha=0.5, label="Holding"),
        mpatches.Patch(color=PHASE_COLORS["cool"][0], alpha=0.5, label="Cooling"),
        pressure_line,
    ]
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # ── Panel 2: GradCAM saliency ──────────────────────────────────────────
    ax2 = axes[1]
    _shade_phases(ax2)
    ax2.fill_between(t, 0, saliency, color="#E74C3C", alpha=0.4)
    ax2.plot(t, saliency, color="#C0392B", linewidth=1.5)
    ax2.set_ylabel("Saliency (0-1)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("GradCAM Temporal Saliency — where the model attended", fontsize=10)
    ax2.grid(alpha=0.3)

    # ── Panel 3: SE channel weights (optional) ─────────────────────────────
    if se_weights_per_block is not None:
        ax3 = axes[2]
        channel_names = [
            "DXP Pressure", "DXP Position", "TCE Temp", "TCN Temp",
            "DOS Rate", "Trig Injection", "Trig Holding", "Trig Cooling",
        ]
        n_channels = len(channel_names)
        n_blocks = len(se_weights_per_block)
        bar_height = 0.8 / n_blocks          # divide available height among blocks
        y_base = np.arange(n_channels)
        block_colors = plt.cm.tab10(np.linspace(0, 0.9, n_blocks))

        for b_idx, (weights, color) in enumerate(zip(se_weights_per_block, block_colors)):
            offsets = y_base + b_idx * bar_height - (n_blocks - 1) * bar_height / 2
            ax3.barh(offsets, weights[:n_channels], height=bar_height * 0.9,
                     color=color, label=f"Block {b_idx + 1}")

        ax3.set_yticks(y_base)
        ax3.set_yticklabels(channel_names, fontsize=8)
        ax3.set_title("SE Channel Weights — which sensors the model weighted per block", fontsize=10)
        ax3.legend(loc="lower right", fontsize=8)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
