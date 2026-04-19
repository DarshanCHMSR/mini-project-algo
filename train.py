"""
Train and evaluate TCN, LSTM, and Logistic Regression baselines.

Improvements for stronger defect detection:
- Strict train-only normalization with fold-safe preprocessing
- Held-out test set for final unbiased reporting
- 5-fold StratifiedKFold on train/val pool
- F1-driven threshold tuning for imbalanced classification
- CV test-time ensembling for TCN and LSTM
"""

from __future__ import annotations

import copy
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
from tcn_model import FocalLoss, LSTMClassifier, TCNDefectClassifier


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
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
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
    _, _, _, tv_stats = preprocess_probays(PARQUET_PATH, train_indices=trainval_indices)
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


if __name__ == "__main__":
    main()
