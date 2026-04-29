"""
GradCAM-based temporal saliency explainer for SE-GradCAM TCN.

Produces timestep-level heatmaps showing where the model attended when making
predictions, aligned to injection molding process phases. Causally valid because
the TCN uses left-only padding.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class GradCAMExplainer:
    """
    Gradient-weighted Class Activation Mapping for TCN temporal saliency.

    Hooks into the final TCNBlock to capture feature maps and gradients,
    producing per-timestep saliency scores that indicate which time points
    drove the model's prediction.
    """

    def __init__(self, model):
        """
        Initialize explainer with a trained TCNDefectClassifier.

        Args:
            model: Trained TCNDefectClassifier instance.
        """
        self.model = model
        self.feature_map = None  # Captured output from final TCNBlock
        self.gradient = None     # Captured gradient w.r.t feature map
        self._forward_handle = None   # Hook handle for forward pass
        self._backward_handle = None  # Hook handle for backward pass

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the final TCNBlock."""
        # Access the final TCNBlock (last element of sequential)
        last_block = self.model.tcn_blocks[-1]

        def forward_hook(module, input, output):
            """Capture the output feature map during forward pass."""
            self.feature_map = output.detach()

        def backward_hook(module, grad_input, grad_output):
            """Capture the gradient w.r.t. the feature map during backward."""
            self.gradient = grad_output[0].detach()

        # Register hooks and store handles for cleanup
        self._forward_handle = last_block.register_forward_hook(forward_hook)
        self._backward_handle = last_block.register_full_backward_hook(backward_hook)

    def _remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def explain(self, x_ts: torch.Tensor, x_scalar: torch.Tensor) -> np.ndarray:
        """
        Compute GradCAM saliency for a single batch of samples.

        Steps:
        1) Register hooks on final TCNBlock
        2) Forward pass to capture feature maps
        3) Backward pass to capture gradients
        4) Compute gradient-weighted channel combination
        5) Normalize to [0, 1] per sample
        6) Remove hooks and return

        Args:
            x_ts: Time-series input (batch, channels, timesteps)
            x_scalar: Scalar features input (batch, n_scalar)

        Returns:
            Numpy array of shape (batch, timesteps) with saliency in [0, 1].
        """
        # Register hooks
        self._register_hooks()

        # Set model to eval mode to disable dropout
        self.model.eval()
        # Clear any existing gradients
        self.model.zero_grad()

        # Forward pass and immediate backward to compute gradients
        logit = self.model(x_ts, x_scalar)
        logit.backward()

        # Remove hooks to free memory
        self._remove_hooks()

        # Compute per-timestep saliency via gradient-weighted channel combination
        # gradient shape: (batch, channels, timesteps)
        # Average gradient over the channel dimension
        weights = self.gradient.mean(dim=1)  # (batch, timesteps)

        # Multiply weights by feature maps and sum over channels
        # feature_map shape: (batch, channels, timesteps)
        cam = (weights.unsqueeze(1) * self.feature_map).sum(dim=1)  # (batch, timesteps)

        # Apply ReLU to keep only positive contributions (attending regions)
        cam = F.relu(cam)

        # Normalize per sample to [0, 1] along time axis
        cam_min = cam.min(dim=-1, keepdim=True).values
        cam_max = cam.max(dim=-1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Return as numpy, squeeze batch dimension for single sample
        return cam.squeeze(0).cpu().numpy()


def plot_saliency_overlay(
    raw_pressure: np.ndarray,
    saliency: np.ndarray,
    trig_inj: np.ndarray,
    trig_hld: np.ndarray,
    trig_cool: np.ndarray,
    lbl_nok: int,
    cycle_id: str,
    se_weights_per_block: list | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot GradCAM saliency overlaid with pressure signal and process phases.

    Creates multi-panel visualization showing:
    - Panel 1: Raw pressure with phase backgrounds
    - Panel 2: GradCAM saliency heatmap
    - Panel 3 (optional): SE channel weights per block

    Args:
        raw_pressure: DXP_Inj1PrsAct values (bar), shape (T,)
        saliency: GradCAM saliency [0-1], shape (T,)
        trig_inj: Injection phase boolean flags, shape (T,)
        trig_hld: Holding phase boolean flags, shape (T,)
        trig_cool: Cooling phase boolean flags, shape (T,)
        lbl_nok: True label (0=OK, 1=DEFECTIVE)
        cycle_id: Cycle identifier string for plot title
        se_weights_per_block: Optional list of SE weights, one per block, each shape (n_channels,)
        save_path: Optional file path to save figure (e.g., "plot.png")
    """
    # Determine number of subplots: 2 if no SE weights, 3 with SE weights
    n_rows = 3 if se_weights_per_block is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=True)

    if n_rows == 2:
        ax_pressure, ax_saliency = axes
    else:
        ax_pressure, ax_saliency, ax_se = axes

    # Convert timestep indices to seconds (200 Hz sampling = 0.005 s per step)
    time_seconds = np.arange(len(raw_pressure)) * 0.005

    # Define phase colors and labels for legend
    phase_colors = {
        'injection': '#3498DB',    # blue
        'holding': '#E67E22',      # orange
        'cooling': '#27AE60',      # green
    }

    # Helper function to add phase background shading
    def add_phase_backgrounds(ax, time_s, trig_i, trig_h, trig_c):
        """Add fill_between backgrounds for each phase."""
        # Injection phase background
        ax.fill_between(
            time_s,
            0, 1,
            where=trig_i > 0.5,
            color=phase_colors['injection'],
            alpha=0.12,
            transform=ax.get_xaxis_transform(),
            label='_Injection',
        )
        # Holding phase background
        ax.fill_between(
            time_s,
            0, 1,
            where=trig_h > 0.5,
            color=phase_colors['holding'],
            alpha=0.12,
            transform=ax.get_xaxis_transform(),
            label='_Holding',
        )
        # Cooling phase background
        ax.fill_between(
            time_s,
            0, 1,
            where=trig_c > 0.5,
            color=phase_colors['cooling'],
            alpha=0.12,
            transform=ax.get_xaxis_transform(),
            label='_Cooling',
        )

    # =========================================================================
    # Panel 1: Pressure signal with phase backgrounds
    # =========================================================================
    add_phase_backgrounds(ax_pressure, time_seconds, trig_inj, trig_hld, trig_cool)
    ax_pressure.plot(time_seconds, raw_pressure, color='#2C3E50', linewidth=1.2, label='Pressure')
    ax_pressure.set_ylabel('Pressure (bar)')
    ax_pressure.grid(alpha=0.3)

    # Create legend with phase patches + pressure line
    phase_patches = [
        Patch(facecolor=phase_colors['injection'], alpha=0.12, label='Injection'),
        Patch(facecolor=phase_colors['holding'], alpha=0.12, label='Holding'),
        Patch(facecolor=phase_colors['cooling'], alpha=0.12, label='Cooling'),
    ]
    ax_pressure.legend(handles=phase_patches, loc='upper right')

    # =========================================================================
    # Panel 2: GradCAM saliency
    # =========================================================================
    add_phase_backgrounds(ax_saliency, time_seconds, trig_inj, trig_hld, trig_cool)
    ax_saliency.fill_between(
        time_seconds,
        saliency,
        color='#E74C3C',
        alpha=0.4,
        label='Saliency'
    )
    ax_saliency.plot(time_seconds, saliency, color='#C0392B', linewidth=1.5)
    ax_saliency.set_ylabel('Saliency (0-1)')
    ax_saliency.set_ylim(0, 1.05)
    ax_saliency.set_title('GradCAM Temporal Saliency — where the model attended')
    ax_saliency.grid(alpha=0.3)

    # =========================================================================
    # Panel 3: SE channel weights (if provided)
    # =========================================================================
    if se_weights_per_block is not None and len(se_weights_per_block) > 0:
        channel_names = [
            'DXP Pressure',
            'DXP Position',
            'TCE Temp',
            'TCN Temp',
            'DOS Rate',
            'Trig Injection',
            'Trig Holding',
            'Trig Cooling',
        ]

        # Prepare data for grouped bar chart
        n_channels = len(channel_names)
        n_blocks = len(se_weights_per_block)
        x_pos = np.arange(n_channels)
        bar_width = 0.8 / n_blocks  # divide width equally among blocks

        # Color palette for blocks
        colors = plt.cm.Set3(np.linspace(0, 1, n_blocks))

        for block_idx, se_weights in enumerate(se_weights_per_block):
            # se_weights shape: (batch, n_channels) or (n_channels,)
            if se_weights.ndim == 2:
                se_weights = se_weights[0]  # Take first sample
            se_weights_np = se_weights.cpu().numpy() if torch.is_tensor(se_weights) else se_weights

            offset = (block_idx - n_blocks / 2 + 0.5) * bar_width
            ax_se.barh(
                x_pos + offset,
                se_weights_np,
                bar_width,
                label=f'Block {block_idx + 1}',
                color=colors[block_idx],
            )

        ax_se.set_yticks(x_pos)
        ax_se.set_yticklabels(channel_names)
        ax_se.set_xlabel('SE Weight')
        ax_se.set_title('SE Channel Weights — which sensors the model weighted per block')
        ax_se.legend(loc='lower right')
        ax_se.grid(alpha=0.3, axis='x')

    # =========================================================================
    # Figure title with label coloring
    # =========================================================================
    if lbl_nok == 1:
        label_text = 'DEFECTIVE (LBL_NOK=1)'
        label_color = 'red'
    else:
        label_text = 'OK (LBL_NOK=0)'
        label_color = 'green'

    fig.suptitle(
        f'Cycle {cycle_id} — {label_text}',
        fontsize=14,
        fontweight='bold',
        color=label_color,
    )

    # Set x-axis label only on the bottom panel
    if n_rows == 2:
        ax_saliency.set_xlabel('Time (seconds)')
    else:
        ax_se.set_xlabel('Time (seconds)')

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
