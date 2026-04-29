"""
Standalone script to generate saliency explanation for a single injection cycle.

Usage:
    python explain_single_cycle.py <MachineCycleID>

Example:
    python explain_single_cycle.py "CYCLE_12345"
"""

from __future__ import annotations

import pickle
import sys
import os
from pathlib import Path

import numpy as np
import torch

from preprocess import preprocess_probays, load_raw_dataframe
from tcn_model import TCNDefectClassifier
from explainer import GradCAMExplainer, plot_saliency_overlay


# Configuration matching train.py
IN_CHANNELS = 8
CHANNEL_LIST = [64, 64, 128]
KERNEL_SIZE = 3
N_SCALAR_FEATS = 5
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """Load cycle data, run GradCAM explainer, generate saliency plot."""
    if len(sys.argv) < 2:
        print("Usage: python explain_single_cycle.py <MachineCycleID>")
        sys.exit(1)

    cycle_id_arg = sys.argv[1]

    # Step 1: Load parquet and find the cycle
    print(f"Loading dataset...")
    df = load_raw_dataframe("dataset_V2.parquet")

    # Find row matching the cycle ID
    if "MET_MachineCycleID" not in df.columns:
        print("ERROR: MET_MachineCycleID column not found in dataset.")
        sys.exit(1)

    df_cycle = df[df["MET_MachineCycleID"].astype(str) == cycle_id_arg]
    if len(df_cycle) == 0:
        print(f"ERROR: Cycle {cycle_id_arg} not found in dataset.")
        sys.exit(1)

    cycle_idx = df_cycle.index[0]
    print(f"Found cycle at index {cycle_idx}")

    # Step 2: Load normalization statistics
    script_dir = os.path.dirname(os.path.abspath(__file__))
    norm_stats_path = os.path.join(script_dir, "norm_stats.pkl")
    if not os.path.exists(norm_stats_path):
        print(f"ERROR: norm_stats.pkl not found at {norm_stats_path}")
        print("Please run train.py first to generate normalization statistics.")
        sys.exit(1)

    with open(norm_stats_path, "rb") as f:
        norm_stats = pickle.load(f)
    print(f"Loaded normalization statistics from {norm_stats_path}")

    # Step 3: Preprocess the single cycle
    x_ts, x_scalar, y, _ = preprocess_probays(
        "dataset_V2.parquet",
        norm_stats=norm_stats,
    )
    x_ts_sample = x_ts[cycle_idx:cycle_idx+1].to(DEVICE)
    x_scalar_sample = x_scalar[cycle_idx:cycle_idx+1].to(DEVICE)
    y_true = int(y[cycle_idx].item())

    # Enable gradients for explainability
    x_ts_sample.requires_grad_(True)

    # Step 4: Load trained model weights
    model_path = os.path.join(script_dir, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: best_model.pt not found at {model_path}")
        print("Please run train.py first to generate the trained model.")
        sys.exit(1)

    model = TCNDefectClassifier(
        in_channels=IN_CHANNELS,
        channel_list=CHANNEL_LIST,
        kernel_size=KERNEL_SIZE,
        n_scalar_feats=N_SCALAR_FEATS,
        dropout=DROPOUT,
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Loaded model from {model_path}")

    # Step 5: Run GradCAM explainer
    print(f"Generating GradCAM saliency for cycle {cycle_id_arg}...")
    explainer = GradCAMExplainer(model)
    saliency = explainer.explain(x_ts_sample, x_scalar_sample)

    # Step 6: Get SE weights
    se_weights_list = model.get_se_weights()
    se_weights_np = []
    if se_weights_list:
        for weight_tensor in se_weights_list:
            if weight_tensor is not None:
                w = weight_tensor[0].numpy() if weight_tensor.ndim > 1 else weight_tensor.numpy()
                se_weights_np.append(w)

    # Step 7: Load raw time-series data
    raw_cache_path = os.path.join(script_dir, "raw_ts_cache.pkl")
    if os.path.exists(raw_cache_path):
        with open(raw_cache_path, "rb") as f:
            raw_ts_cache = pickle.load(f)
        if cycle_id_arg in raw_ts_cache:
            cache_entry = raw_ts_cache[cycle_id_arg]
            raw_pressure = cache_entry["DXP_Inj1PrsAct"]
            trig_inj = cache_entry["DXP_TrigInj1"]
            trig_hld = cache_entry["DXP_TrigHld1"]
            trig_cool = cache_entry["DXP_TrigCool"]
        else:
            print(f"WARNING: Cycle {cycle_id_arg} not in raw_ts_cache, using zeros")
            raw_pressure = np.zeros(len(saliency), dtype=np.float32)
            trig_inj = np.zeros(len(saliency), dtype=np.float32)
            trig_hld = np.zeros(len(saliency), dtype=np.float32)
            trig_cool = np.zeros(len(saliency), dtype=np.float32)
    else:
        print("WARNING: raw_ts_cache.pkl not found, using zero arrays")
        raw_pressure = np.zeros(len(saliency), dtype=np.float32)
        trig_inj = np.zeros(len(saliency), dtype=np.float32)
        trig_hld = np.zeros(len(saliency), dtype=np.float32)
        trig_cool = np.zeros(len(saliency), dtype=np.float32)

    # Step 8: Get model prediction
    with torch.no_grad():
        pred_logit = model(x_ts_sample, x_scalar_sample).item()
        pred_prob = torch.sigmoid(torch.tensor(pred_logit)).item()

    # Determine prediction threshold (use default 0.5 since we don't have the fold threshold)
    threshold = 0.5
    pred_class = "DEFECTIVE" if pred_prob >= threshold else "OK"

    # Step 9: Generate and save saliency plot
    plot_path = f"saliency_cycle_{cycle_id_arg}.png"
    plot_saliency_overlay(
        raw_pressure=raw_pressure,
        saliency=saliency,
        trig_inj=trig_inj,
        trig_hld=trig_hld,
        trig_cool=trig_cool,
        lbl_nok=y_true,
        cycle_id=cycle_id_arg,
        se_weights_per_block=se_weights_np if se_weights_np else None,
        save_path=plot_path,
    )

    # Step 10: Print summary
    print("\n" + "=" * 70)
    print("EXPLANATION SUMMARY")
    print("=" * 70)
    print(f"Cycle ID: {cycle_id_arg}")
    print(f"True Label: {'DEFECTIVE' if y_true == 1 else 'OK'}")
    print(f"Model Prediction: {pred_prob:.4f}")
    print(f"Predicted Class: {pred_class}")
    print(f"Saliency plot saved to: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
