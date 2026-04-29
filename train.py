from __future__ import annotations

import copy
import pickle
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from preprocess import TARGET_COL, load_raw_dataframe, preprocess_probays
from tcn_model import AsymmetricFocalLoss, LSTMClassifier, TCNDefectClassifier
from explainer import GradCAMExplainer, plot_saliency_overlay


# -----------------------------
# Configuration (requested)
# -----------------------------

PARQUET_PATH = "dataset_V2.parquet"

IN_CHANNELS = 8
CHANNEL_LIST = [64, 64, 128]
KERNEL_SIZE = 3
N_SCALAR_FEATS = 5
DROPOUT = 0.1

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
GRAD_CLIP_NORM = 1.0
LR_PATIENCE = 5
LR_FACTOR = 0.5

N_SPLITS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class FoldArtifacts:
    model_state: Dict
    norm_stats: Dict
    threshold: float


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    x_ts: torch.Tensor,
    x_scalar: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    train: bool,
) -> DataLoader:
    """Build DataLoader with weighted sampling for imbalanced train batches."""
    ds = TensorDataset(x_ts, x_scalar, y)
    if not train:
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    y_np = y.cpu().numpy().astype(int)
    class_counts = np.bincount(y_np, minlength=2)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_np]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Choose threshold maximizing F1 on validation/OOF probabilities."""
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 181):
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    """Compute requested metrics with positive class = LBL_NOK=1."""
    preds = (y_prob >= threshold).astype(int)

    metrics = {
        "auc_roc": 0.0,
        "auc_pr": 0.0,
        "f1": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "threshold": float(threshold),
    }

    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        pass

    try:
        metrics["auc_pr"] = average_precision_score(y_true, y_prob)
    except ValueError:
        pass

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    metrics["confusion_matrix"] = cm
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])
    metrics["classification_report"] = classification_report(
        y_true,
        preds,
        labels=[0, 1],
        target_names=["OK (0)", "NOK (1)"],
        zero_division=0,
    )
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    losses: List[float] = []

    for x_ts, x_sc, y in loader:
        x_ts = x_ts.to(DEVICE)
        x_sc = x_sc.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_ts, x_sc)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def infer_probs(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    all_probs: List[np.ndarray] = []
    for x_ts, x_sc, _ in loader:
        x_ts = x_ts.to(DEVICE)
        x_sc = x_sc.to(DEVICE)
        logits = model(x_ts, x_sc)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
    if not all_probs:
        return np.array([], dtype=np.float32)
    return np.concatenate(all_probs).astype(np.float32)


def fit_neural_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val: np.ndarray,
) -> Tuple[nn.Module, float, np.ndarray]:
    """Train neural model with requested optimizer/loss/scheduler/stopping."""
    # Use AsymmetricFocalLoss with higher gamma_pos to penalize false negatives more
    criterion = AsymmetricFocalLoss(gamma_neg=2.0, gamma_pos=4.0, alpha=0.75, clip=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=LR_PATIENCE,
        factor=LR_FACTOR,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_probs = np.zeros_like(y_val, dtype=np.float32)
    patience = 0

    for epoch in range(MAX_EPOCHS):
        _ = train_one_epoch(model, train_loader, criterion, optimizer)

        # Validation monitoring is on F1 as requested.
        val_probs = infer_probs(model, val_loader)
        val_threshold = find_best_threshold(y_val, val_probs)
        val_preds = (val_probs >= val_threshold).astype(int)
        val_f1 = f1_score(y_val, val_preds, pos_label=1, zero_division=0)

        # Scheduler is requested on min validation loss; use focal loss estimate.
        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for x_ts, x_sc, y_batch in val_loader:
                x_ts = x_ts.to(DEVICE)
                x_sc = x_sc.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                logits = model(x_ts, x_sc)
                val_losses.append(float(criterion(logits, y_batch).item()))
        scheduler.step(float(np.mean(val_losses)) if val_losses else 0.0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            best_probs = val_probs.copy()
            patience = 0
        else:
            patience += 1

        if patience >= EARLY_STOP_PATIENCE:
            break

    model.load_state_dict(best_state)
    best_threshold = find_best_threshold(y_val, best_probs)
    return model, best_threshold, best_probs


def cv_train_tcn(
    parquet_path: str,
    trainval_indices: np.ndarray,
    labels_all: np.ndarray,
    t_max_trainval: int,
) -> Tuple[List[FoldArtifacts], np.ndarray, np.ndarray]:
    """5-fold CV training on train/val pool for TCN and OOF collection."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    labels_tv = labels_all[trainval_indices]
    oof_probs = np.zeros(len(trainval_indices), dtype=np.float32)
    fold_artifacts: List[FoldArtifacts] = []

    for fold, (inner_tr, inner_va) in enumerate(skf.split(np.arange(len(trainval_indices)), labels_tv), start=1):
        tr_abs = trainval_indices[inner_tr]
        va_abs = trainval_indices[inner_va]

        x_ts, x_scalar, y, norm_stats = preprocess_probays(
            parquet_path,
            train_indices=tr_abs,
            t_max=t_max_trainval,
        )

        x_ts_tr, x_sc_tr, y_tr = x_ts[tr_abs], x_scalar[tr_abs], y[tr_abs]
        x_ts_va, x_sc_va, y_va = x_ts[va_abs], x_scalar[va_abs], y[va_abs]

        train_loader = make_loader(x_ts_tr, x_sc_tr, y_tr, BATCH_SIZE, train=True)
        val_loader = make_loader(x_ts_va, x_sc_va, y_va, BATCH_SIZE, train=False)

        model = TCNDefectClassifier(
            in_channels=IN_CHANNELS,
            channel_list=CHANNEL_LIST,
            kernel_size=KERNEL_SIZE,
            n_scalar_feats=N_SCALAR_FEATS,
            dropout=DROPOUT,
        ).to(DEVICE)

        # Requested derivation comment:
        # total receptive field = 1 + (kernel_size - 1) * sum(all dilation values across all blocks)
        # Here dilations are [1, 2, 4] for 3 blocks, so RF = 1 + (3 - 1) * (1 + 2 + 4) = 15.

        model, fold_threshold, fold_val_probs = fit_neural_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            y_val=y_va.numpy().astype(int),
        )

        oof_probs[inner_va] = fold_val_probs
        fold_artifacts.append(
            FoldArtifacts(
                model_state=copy.deepcopy(model.state_dict()),
                norm_stats=norm_stats,
                threshold=float(fold_threshold),
            )
        )

        fold_metrics = compute_metrics(
            y_true=y_va.numpy().astype(int),
            y_prob=fold_val_probs,
            threshold=fold_threshold,
        )
        print(
            f"Fold {fold}/{N_SPLITS} | "
            f"AUC-ROC={fold_metrics['auc_roc']:.4f} "
            f"AUC-PR={fold_metrics['auc_pr']:.4f} "
            f"F1={fold_metrics['f1']:.4f} "
            f"thr={fold_threshold:.3f}"
        )

    oof_y = labels_tv.astype(int)
    return fold_artifacts, oof_probs, oof_y


def ensemble_predict_tcn(
    parquet_path: str,
    test_indices: np.ndarray,
    fold_artifacts: List[FoldArtifacts],
    t_max_trainval: int,
) -> np.ndarray:
    """Average test probabilities from all CV TCN fold models."""
    all_fold_test_probs: List[np.ndarray] = []

    for artifact in fold_artifacts:
        x_ts, x_sc, y, _ = preprocess_probays(
            parquet_path,
            norm_stats=artifact.norm_stats,
            t_max=t_max_trainval,
        )

        x_ts_te = x_ts[test_indices]
        x_sc_te = x_sc[test_indices]
        y_te = y[test_indices]

        test_loader = make_loader(x_ts_te, x_sc_te, y_te, BATCH_SIZE, train=False)

        model = TCNDefectClassifier(
            in_channels=IN_CHANNELS,
            channel_list=CHANNEL_LIST,
            kernel_size=KERNEL_SIZE,
            n_scalar_feats=N_SCALAR_FEATS,
            dropout=DROPOUT,
        ).to(DEVICE)
        model.load_state_dict(artifact.model_state)

        probs = infer_probs(model, test_loader)
        all_fold_test_probs.append(probs)

    return np.mean(np.stack(all_fold_test_probs, axis=0), axis=0)


def cv_train_lstm(
    parquet_path: str,
    trainval_indices: np.ndarray,
    labels_all: np.ndarray,
    t_max_trainval: int,
) -> Tuple[List[FoldArtifacts], np.ndarray, np.ndarray]:
    """5-fold CV for LSTM baseline using same data pipeline and protocol."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    labels_tv = labels_all[trainval_indices]
    oof_probs = np.zeros(len(trainval_indices), dtype=np.float32)
    fold_artifacts: List[FoldArtifacts] = []

    for fold, (inner_tr, inner_va) in enumerate(skf.split(np.arange(len(trainval_indices)), labels_tv), start=1):
        tr_abs = trainval_indices[inner_tr]
        va_abs = trainval_indices[inner_va]

        x_ts, x_scalar, y, norm_stats = preprocess_probays(
            parquet_path,
            train_indices=tr_abs,
            t_max=t_max_trainval,
        )

        x_ts_tr, x_sc_tr, y_tr = x_ts[tr_abs], x_scalar[tr_abs], y[tr_abs]
        x_ts_va, x_sc_va, y_va = x_ts[va_abs], x_scalar[va_abs], y[va_abs]

        train_loader = make_loader(x_ts_tr, x_sc_tr, y_tr, BATCH_SIZE, train=True)
        val_loader = make_loader(x_ts_va, x_sc_va, y_va, BATCH_SIZE, train=False)

        model = LSTMClassifier(
            in_channels=IN_CHANNELS,
            hidden_size=128,
            num_layers=2,
            n_scalar_feats=N_SCALAR_FEATS,
            dropout=DROPOUT,
        ).to(DEVICE)

        model, fold_threshold, fold_val_probs = fit_neural_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            y_val=y_va.numpy().astype(int),
        )

        oof_probs[inner_va] = fold_val_probs
        fold_artifacts.append(
            FoldArtifacts(
                model_state=copy.deepcopy(model.state_dict()),
                norm_stats=norm_stats,
                threshold=float(fold_threshold),
            )
        )

        fold_metrics = compute_metrics(
            y_true=y_va.numpy().astype(int),
            y_prob=fold_val_probs,
            threshold=fold_threshold,
        )
        print(
            f"LSTM Fold {fold}/{N_SPLITS} | "
            f"AUC-ROC={fold_metrics['auc_roc']:.4f} "
            f"AUC-PR={fold_metrics['auc_pr']:.4f} "
            f"F1={fold_metrics['f1']:.4f} "
            f"thr={fold_threshold:.3f}"
        )

    oof_y = labels_tv.astype(int)
    return fold_artifacts, oof_probs, oof_y


def ensemble_predict_lstm(
    parquet_path: str,
    test_indices: np.ndarray,
    fold_artifacts: List[FoldArtifacts],
    t_max_trainval: int,
) -> np.ndarray:
    """Average test probabilities from all CV LSTM fold models."""
    all_fold_test_probs: List[np.ndarray] = []

    for artifact in fold_artifacts:
        x_ts, x_sc, y, _ = preprocess_probays(
            parquet_path,
            norm_stats=artifact.norm_stats,
            t_max=t_max_trainval,
        )

        x_ts_te = x_ts[test_indices]
        x_sc_te = x_sc[test_indices]
        y_te = y[test_indices]

        test_loader = make_loader(x_ts_te, x_sc_te, y_te, BATCH_SIZE, train=False)

        model = LSTMClassifier(
            in_channels=IN_CHANNELS,
            hidden_size=128,
            num_layers=2,
            n_scalar_feats=N_SCALAR_FEATS,
            dropout=DROPOUT,
        ).to(DEVICE)
        model.load_state_dict(artifact.model_state)

        probs = infer_probs(model, test_loader)
        all_fold_test_probs.append(probs)

    return np.mean(np.stack(all_fold_test_probs, axis=0), axis=0)


def train_logreg_baseline(
    parquet_path: str,
    trainval_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train logistic regression baseline on scalar features only."""
    x_ts, x_scalar, y, _ = preprocess_probays(
        parquet_path,
        train_indices=trainval_indices,
    )
    del x_ts

    x_train = x_scalar[trainval_indices].numpy()
    y_train = y[trainval_indices].numpy().astype(int)
    x_test = x_scalar[test_indices].numpy()
    y_test = y[test_indices].numpy().astype(int)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)

    train_prob = clf.predict_proba(x_train)[:, 1]
    test_prob = clf.predict_proba(x_test)[:, 1]

    train_thr = find_best_threshold(y_train, train_prob)
    return test_prob, np.array([train_thr], dtype=np.float32)


def print_metrics(title: str, metrics: Dict) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"F1:      {metrics['f1']:.4f}")
    print(f"Threshold: {metrics['threshold']:.3f}")
    print("Confusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(metrics["classification_report"])


def main() -> None:
    set_seed()
    print(f"Device: {DEVICE}")

    df = load_raw_dataframe(PARQUET_PATH)
    y_all = df[TARGET_COL].astype(int).values
    all_indices = np.arange(len(df))

    # Held-out test split for final reporting.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(sss.split(all_indices, y_all))
    trainval_indices = all_indices[trainval_idx]
    test_indices = all_indices[test_idx]

    print(
        f"Samples: total={len(df)} trainval={len(trainval_indices)} test={len(test_indices)} | "
        f"NOK ratio total={y_all.mean():.3f}"
    )

    # Fix T_max using only trainval pool (no test leakage).
    # Call with save_artifacts=True to save norm_stats.pkl and raw_ts_cache.pkl
    _, _, _, tv_stats = preprocess_probays(PARQUET_PATH, train_indices=trainval_indices, save_artifacts=True)
    t_max_trainval = int(tv_stats["t_max"])
    print(f"Using T_max={t_max_trainval} computed from trainval subset only")

    print("\nTraining TCN with 5-fold CV...")
    tcn_folds, tcn_oof_probs, tcn_oof_y = cv_train_tcn(
        parquet_path=PARQUET_PATH,
        trainval_indices=trainval_indices,
        labels_all=y_all,
        t_max_trainval=t_max_trainval,
    )
    tcn_final_thr = find_best_threshold(tcn_oof_y, tcn_oof_probs)
    tcn_test_probs = ensemble_predict_tcn(
        parquet_path=PARQUET_PATH,
        test_indices=test_indices,
        fold_artifacts=tcn_folds,
        t_max_trainval=t_max_trainval,
    )

    print("\nTraining LSTM baseline with 5-fold CV...")
    lstm_folds, lstm_oof_probs, lstm_oof_y = cv_train_lstm(
        parquet_path=PARQUET_PATH,
        trainval_indices=trainval_indices,
        labels_all=y_all,
        t_max_trainval=t_max_trainval,
    )
    lstm_final_thr = find_best_threshold(lstm_oof_y, lstm_oof_probs)
    lstm_test_probs = ensemble_predict_lstm(
        parquet_path=PARQUET_PATH,
        test_indices=test_indices,
        fold_artifacts=lstm_folds,
        t_max_trainval=t_max_trainval,
    )

    print("\nTraining Logistic Regression baseline...")
    logreg_test_probs, logreg_thr_arr = train_logreg_baseline(
        parquet_path=PARQUET_PATH,
        trainval_indices=trainval_indices,
        test_indices=test_indices,
    )
    logreg_final_thr = float(logreg_thr_arr[0])

    y_test = y_all[test_indices].astype(int)

    tcn_metrics = compute_metrics(y_true=y_test, y_prob=tcn_test_probs, threshold=tcn_final_thr)
    lstm_metrics = compute_metrics(y_true=y_test, y_prob=lstm_test_probs, threshold=lstm_final_thr)
    logreg_metrics = compute_metrics(y_true=y_test, y_prob=logreg_test_probs, threshold=logreg_final_thr)

    print_metrics("TCN Held-out Test Metrics", tcn_metrics)
    print_metrics("LSTM Held-out Test Metrics", lstm_metrics)
    print_metrics("Logistic Regression Held-out Test Metrics", logreg_metrics)

    results = pd.DataFrame(
        [
            {
                "Model": "TCN",
                "AUC-ROC": tcn_metrics["auc_roc"],
                "AUC-PR": tcn_metrics["auc_pr"],
                "F1 (NOK=1)": tcn_metrics["f1"],
            },
            {
                "Model": "LSTM",
                "AUC-ROC": lstm_metrics["auc_roc"],
                "AUC-PR": lstm_metrics["auc_pr"],
                "F1 (NOK=1)": lstm_metrics["f1"],
            },
            {
                "Model": "Logistic Regression",
                "AUC-ROC": logreg_metrics["auc_roc"],
                "AUC-PR": logreg_metrics["auc_pr"],
                "F1 (NOK=1)": logreg_metrics["f1"],
            },
        ]
    )

    print("\n" + "=" * 70)
    print("Model Comparison on Held-out Test Set")
    print("=" * 70)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Save the best TCN model from the first fold for explain_single_cycle.py to use
    best_model_path = "best_model.pt"
    best_tcn = TCNDefectClassifier(
        in_channels=IN_CHANNELS,
        channel_list=CHANNEL_LIST,
        kernel_size=KERNEL_SIZE,
        n_scalar_feats=N_SCALAR_FEATS,
        dropout=DROPOUT,
    ).to(DEVICE)
    best_tcn.load_state_dict(tcn_folds[0].model_state)
    torch.save(best_tcn.state_dict(), best_model_path)
    print(f"\nSaved best TCN model to {best_model_path}")

    # =========================================================================
    # EXPLAINABILITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPLAINABILITY ANALYSIS — GradCAM Temporal Saliency")
    print("=" * 70)

    # Load raw data for saliency analysis
    df_full = load_raw_dataframe(PARQUET_PATH)
    x_ts_full, x_scalar_full, y_full, norm_stats_final = preprocess_probays(
        PARQUET_PATH,
        train_indices=trainval_indices,
    )

    # Load raw time-series cache (needed for pressure and trigger arrays)
    raw_cache_path = os.path.join(os.path.dirname(__file__), "raw_ts_cache.pkl")
    if os.path.exists(raw_cache_path):
        with open(raw_cache_path, "rb") as f:
            raw_ts_cache = pickle.load(f)
    else:
        raw_ts_cache = {}

    # Identify all defective cycles in the test set
    y_test_full = y_all[test_indices].astype(int)
    defective_mask = y_test_full == 1
    defective_test_indices_local = np.where(defective_mask)[0]
    defective_test_indices_global = test_indices[defective_test_indices_local]

    # Create a best model from the first fold for explainability
    best_tcn_fold = tcn_folds[0]
    model_explain = TCNDefectClassifier(
        in_channels=IN_CHANNELS,
        channel_list=CHANNEL_LIST,
        kernel_size=KERNEL_SIZE,
        n_scalar_feats=N_SCALAR_FEATS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model_explain.load_state_dict(best_tcn_fold.model_state)

    # Get cycle IDs from dataframe
    cycle_ids = df_full["MET_MachineCycleID"].astype(str).values if "MET_MachineCycleID" in df_full.columns else [str(i) for i in range(len(df_full))]

    # Saliency peak times and phase-wise statistics
    saliency_peak_times_sec = []
    saliency_inj_means = []
    saliency_hld_means = []
    saliency_cool_means = []

    for local_idx, global_idx in enumerate(defective_test_indices_global):
        cycle_id = cycle_ids[global_idx]

        # Get sample tensors
        x_ts_sample = x_ts_full[global_idx:global_idx+1].to(DEVICE)
        x_scalar_sample = x_scalar_full[global_idx:global_idx+1].to(DEVICE)
        # Enable gradients for explainability
        x_ts_sample.requires_grad_(True)

        # Instantiate explainer
        explainer = GradCAMExplainer(model_explain)
        saliency = explainer.explain(x_ts_sample, x_scalar_sample)

        # Get SE weights from the model (after explain call)
        se_weights_list = model_explain.get_se_weights()
        se_weights_np = []
        if se_weights_list:
            for weight_tensor in se_weights_list:
                if weight_tensor is not None:
                    w = weight_tensor[0].numpy() if weight_tensor.ndim > 1 else weight_tensor.numpy()
                    se_weights_np.append(w)

        # Retrieve raw data for visualization
        raw_pressure = np.zeros(len(saliency), dtype=np.float32)
        trig_inj = np.zeros(len(saliency), dtype=np.float32)
        trig_hld = np.zeros(len(saliency), dtype=np.float32)
        trig_cool = np.zeros(len(saliency), dtype=np.float32)

        if cycle_id in raw_ts_cache:
            cache_entry = raw_ts_cache[cycle_id]
            raw_pressure = cache_entry["DXP_Inj1PrsAct"]
            trig_inj = cache_entry["DXP_TrigInj1"]
            trig_hld = cache_entry["DXP_TrigHld1"]
            trig_cool = cache_entry["DXP_TrigCool"]

        # Generate saliency plot
        save_path = f"saliency_cycle_{cycle_id}.png"
        plot_saliency_overlay(
            raw_pressure=raw_pressure,
            saliency=saliency,
            trig_inj=trig_inj,
            trig_hld=trig_hld,
            trig_cool=trig_cool,
            lbl_nok=int(y_test_full[local_idx]),
            cycle_id=cycle_id,
            se_weights_per_block=se_weights_np if se_weights_np else None,
            save_path=save_path,
        )

        # Get predicted label
        with torch.no_grad():
            pred_logit = model_explain(x_ts_sample, x_scalar_sample).item()
            pred_prob = torch.sigmoid(torch.tensor(pred_logit)).item()
            pred_label = "DEFECTIVE" if pred_prob >= tcn_final_thr else "OK"

        print(f"Saliency plot saved for cycle {cycle_id}, predicted defective={pred_label}, true label=DEFECTIVE")

        # Compute saliency statistics
        peak_idx = np.argmax(saliency)
        peak_time_sec = peak_idx * 0.005
        saliency_peak_times_sec.append(peak_time_sec)

        # Phase-wise mean saliency
        if trig_inj.sum() > 0:
            saliency_inj_means.append(saliency[trig_inj > 0.5].mean())
        if trig_hld.sum() > 0:
            saliency_hld_means.append(saliency[trig_hld > 0.5].mean())
        if trig_cool.sum() > 0:
            saliency_cool_means.append(saliency[trig_cool > 0.5].mean())

    # Print aggregate statistics
    if saliency_peak_times_sec:
        mean_peak_time = np.mean(saliency_peak_times_sec)
        std_peak_time = np.std(saliency_peak_times_sec)
        print(f"\nMean saliency peak time: {mean_peak_time:.4f} ± {std_peak_time:.4f} seconds")

    if saliency_inj_means:
        mean_inj = np.mean(saliency_inj_means)
        print(f"Mean saliency during injection phase: {mean_inj:.4f}")
    else:
        mean_inj = 0.0

    if saliency_hld_means:
        mean_hld = np.mean(saliency_hld_means)
        print(f"Mean saliency during holding phase: {mean_hld:.4f}")
    else:
        mean_hld = 0.0

    if saliency_cool_means:
        mean_cool = np.mean(saliency_cool_means)
        print(f"Mean saliency during cooling phase: {mean_cool:.4f}")
    else:
        mean_cool = 0.0

    # Determine and print dominant phase
    phase_means = {
        "Injection": mean_inj,
        "Holding": mean_hld,
        "Cooling": mean_cool,
    }
    dominant_phase = max(phase_means, key=phase_means.get)
    print(f"The model primarily attended to the {dominant_phase} phase when predicting defects.")


if __name__ == "__main__":
    main()
