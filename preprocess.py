"""
Preprocessing pipeline for ProBayes defect prediction.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


TIME_SERIES_COLS = [
    "DXP_Inj1PrsAct",
    "DXP_Inj1PosAct",
    "TCE_TemperatureMainLine",
    "TCN_TemperatureMainLine",
    "DOS_acComp1DosRate",
    "DXP_TrigInj1",
    "DXP_TrigHld1",
    "DXP_TrigCool",
]

SCALAR_COLS = [
    "QUA_InjectionPressureMax",
    "QUA_CavityPressureMax",
    "QUA_CycleTime",
    "material_is_PP",
    "ENV_AirTemperature",
]

TARGET_COL = "LBL_NOK"
MATERIAL_COL = "MET_MaterialName"
CYCLE_TRIGGER_COL = "DXP_TrigClpCls"
EPS = 1e-8


def load_raw_dataframe(parquet_path: str) -> pd.DataFrame:
    """Load rows with non-null target labels only."""
    df = pd.read_parquet(parquet_path)
    return df[df[TARGET_COL].notna()].reset_index(drop=True)


def _to_1d_float_array(value) -> np.ndarray:
    """Safely coerce cell content to a 1D float32 numpy array."""
    if value is None:
        return np.array([], dtype=np.float32)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.array([], dtype=np.float32)
    arr = arr.astype(np.float32, copy=False)
    return arr


def _last_true_index(trigger_arr: np.ndarray) -> Optional[int]:
    """Return last index where trigger is true/positive, else None."""
    if trigger_arr.size == 0:
        return None
    idx = np.where(trigger_arr.astype(np.float32) > 0.0)[0]
    if idx.size == 0:
        return None
    return int(idx[-1])


def _compute_cycle_length(row: pd.Series) -> int:
    """
    Determine cycle length as the last True index in DXP_TrigClpCls + 1.
    Falls back to DXP_Inj1PrsAct length if trigger is missing/empty.
    """
    trigger = _to_1d_float_array(row.get(CYCLE_TRIGGER_COL))
    last_idx = _last_true_index(trigger)
    if last_idx is not None:
        return last_idx + 1

    fallback = _to_1d_float_array(row.get("DXP_Inj1PrsAct"))
    return max(int(fallback.size), 1)


def _pad_or_truncate_right(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Right-pad with zeros or truncate to target_len."""
    out = np.zeros(target_len, dtype=np.float32)
    if arr.size == 0:
        return out
    n = min(target_len, arr.size)
    out[:n] = arr[:n]
    return out


def _resolve_cavity_pressure_col(df: pd.DataFrame) -> str:
    """Support both possible cavity pressure column names."""
    if "QUA_CavityPressureMax" in df.columns:
        return "QUA_CavityPressureMax"
    if "QUA_CavityPressure1Max" in df.columns:
        return "QUA_CavityPressure1Max"
    raise KeyError("Neither QUA_CavityPressureMax nor QUA_CavityPressure1Max exists.")


def _build_time_series_tensor(df: pd.DataFrame, t_max: int) -> np.ndarray:
    """Create (N, C, T_max) tensor in exact channel order of TIME_SERIES_COLS."""
    n_samples = len(df)
    n_channels = len(TIME_SERIES_COLS)
    x_ts = np.zeros((n_samples, n_channels, t_max), dtype=np.float32)

    for i in range(n_samples):
        row = df.iloc[i]
        for c, col in enumerate(TIME_SERIES_COLS):
            arr = _to_1d_float_array(row.get(col))
            if "DXP_Trig" in col:
                arr = (arr > 0).astype(np.float32)
            x_ts[i, c] = _pad_or_truncate_right(arr, t_max)

    return x_ts


def _build_scalar_matrix(df: pd.DataFrame) -> np.ndarray:
    """Create (N, 5) scalar matrix in exact SCALAR_COLS order."""
    cavity_col = _resolve_cavity_pressure_col(df)

    material_is_pp = (df[MATERIAL_COL].astype(str).str.upper() == "PP").astype(np.float32).values

    x_scalar = np.stack(
        [
            pd.to_numeric(df["QUA_InjectionPressureMax"], errors="coerce").fillna(0.0).astype(np.float32).values,
            pd.to_numeric(df[cavity_col], errors="coerce").fillna(0.0).astype(np.float32).values,
            pd.to_numeric(df["QUA_CycleTime"], errors="coerce").fillna(0.0).astype(np.float32).values,
            material_is_pp,
            pd.to_numeric(df["ENV_AirTemperature"], errors="coerce").fillna(0.0).astype(np.float32).values,
        ],
        axis=1,
    )

    return x_scalar


def _fit_time_series_stats_by_material(
    x_ts: np.ndarray,
    material_is_pp: np.ndarray,
    train_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Fit channel-wise mean/std on training data with separate groups per material.
    stats arrays shape: (2, C), index 0=ABS, 1=PP.
    """
    n_channels = x_ts.shape[1]
    ts_means = np.zeros((2, n_channels), dtype=np.float32)
    ts_stds = np.ones((2, n_channels), dtype=np.float32)

    for m in (0, 1):
        mat_mask = material_is_pp[train_indices] == m
        mat_train_idx = train_indices[mat_mask]
        if mat_train_idx.size == 0:
            mat_train_idx = train_indices

        mat_data = x_ts[mat_train_idx]
        ts_means[m] = mat_data.mean(axis=(0, 2))
        ts_stds[m] = mat_data.std(axis=(0, 2))

    return {"ts_means_by_material": ts_means, "ts_stds_by_material": ts_stds}


def _apply_time_series_norm_by_material(
    x_ts: np.ndarray,
    material_is_pp: np.ndarray,
    ts_means_by_material: np.ndarray,
    ts_stds_by_material: np.ndarray,
) -> np.ndarray:
    """Apply material-grouped channel-wise z-score normalization."""
    x_out = x_ts.copy()
    for m in (0, 1):
        idx = np.where(material_is_pp == m)[0]
        if idx.size == 0:
            continue
        mean = ts_means_by_material[m][None, :, None]
        std = ts_stds_by_material[m][None, :, None]
        x_out[idx] = (x_out[idx] - mean) / (std + EPS)
    return x_out


def _fit_scalar_stats(x_scalar: np.ndarray, train_indices: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit scalar means/std on training indices only."""
    scalar_means = x_scalar[train_indices].mean(axis=0)
    scalar_stds = x_scalar[train_indices].std(axis=0)
    return {"scalar_means": scalar_means.astype(np.float32), "scalar_stds": scalar_stds.astype(np.float32)}


def _apply_scalar_norm(x_scalar: np.ndarray, scalar_means: np.ndarray, scalar_stds: np.ndarray) -> np.ndarray:
    """Apply scalar z-score normalization to all scalar features."""
    return (x_scalar - scalar_means[None, :]) / (scalar_stds[None, :] + EPS)


def preprocess_probays(
    parquet_path: str,
    train_indices: Optional[np.ndarray] = None,
    norm_stats: Optional[Dict] = None,
    t_max: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Full preprocessing pipeline for train-ready tensors.

    Steps:
    1) load parquet
    2) keep labeled rows
    3) fixed-length TS using p5 cycle length (DXP_TrigClpCls)
    4) stack channels in TIME_SERIES_COLS order
    5) channel-wise TS normalization from train split only
    6) scalar feature extraction + normalization from train split only
    7) extract LBL_NOK
    8) cast to float32 torch tensors
    9) return tensors + stats dict
    """
    df = load_raw_dataframe(parquet_path)
    n_samples = len(df)

    if train_indices is None:
        train_indices = np.arange(n_samples, dtype=np.int64)
    else:
        train_indices = np.asarray(train_indices, dtype=np.int64)

    if t_max is None:
        if norm_stats is not None and "t_max" in norm_stats:
            t_max = int(norm_stats["t_max"])
        else:
            cycle_lengths = np.array([_compute_cycle_length(df.iloc[i]) for i in range(n_samples)], dtype=np.int64)
            # Requested formula: use 5th percentile to avoid extreme outlier padding.
            # (This follows the provided project specification exactly.)
            t_max = int(max(np.percentile(cycle_lengths[train_indices], 5), 1))

    x_ts_np = _build_time_series_tensor(df, t_max=t_max)
    x_scalar_np = _build_scalar_matrix(df)
    y_np = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(np.float32).values

    material_is_pp = (df[MATERIAL_COL].astype(str).str.upper() == "PP").astype(np.int64).values

    if norm_stats is None:
        ts_stats = _fit_time_series_stats_by_material(
            x_ts=x_ts_np,
            material_is_pp=material_is_pp,
            train_indices=train_indices,
        )
        scalar_stats = _fit_scalar_stats(x_scalar_np, train_indices=train_indices)

        norm_stats = {
            "t_max": int(t_max),
            "time_series_cols": list(TIME_SERIES_COLS),
            "scalar_cols": list(SCALAR_COLS),
            "ts_means_by_material": ts_stats["ts_means_by_material"],
            "ts_stds_by_material": ts_stats["ts_stds_by_material"],
            "scalar_means": scalar_stats["scalar_means"],
            "scalar_stds": scalar_stats["scalar_stds"],
        }

    x_ts_np = _apply_time_series_norm_by_material(
        x_ts=x_ts_np,
        material_is_pp=material_is_pp,
        ts_means_by_material=np.asarray(norm_stats["ts_means_by_material"], dtype=np.float32),
        ts_stds_by_material=np.asarray(norm_stats["ts_stds_by_material"], dtype=np.float32),
    )

    x_scalar_np = _apply_scalar_norm(
        x_scalar=x_scalar_np,
        scalar_means=np.asarray(norm_stats["scalar_means"], dtype=np.float32),
        scalar_stds=np.asarray(norm_stats["scalar_stds"], dtype=np.float32),
    )

    x_ts = torch.tensor(x_ts_np, dtype=torch.float32)
    x_scalar = torch.tensor(x_scalar_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    return x_ts, x_scalar, y, norm_stats


# Backward-compatible alias for older imports.
def preprocess_probayes(
    parquet_path: str,
    train_indices: Optional[np.ndarray] = None,
    norm_stats: Optional[Dict] = None,
    t_max: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    return preprocess_probays(
        parquet_path=parquet_path,
        train_indices=train_indices,
        norm_stats=norm_stats,
        t_max=t_max,
    )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "dataset_V2.parquet"
    x_ts_, x_sc_, y_, stats_ = preprocess_probays(path)
    print("x_ts:", tuple(x_ts_.shape))
    print("x_scalar:", tuple(x_sc_.shape))
    print("y:", tuple(y_.shape))
    print("t_max:", stats_["t_max"])
