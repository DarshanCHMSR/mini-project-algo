# Implementation Flow (End-to-End)

This document explains how the system works from raw data to final model comparison.

## A. High-Level Flow
1. Read labeled cycles from parquet
2. Build fixed-shape tensors for time-series and scalar features
3. Perform leakage-safe normalization
4. Split data into train/validation pool and held-out test set
5. Train TCN via 5-fold stratified CV
6. Train LSTM baseline via same CV protocol
7. Train logistic regression baseline on scalar features
8. Tune thresholds for positive-class F1
9. Ensemble CV models on held-out test
10. Compute final metrics and comparison table

---

## B. Data Ingestion and Shaping
File: `preprocess.py`

### B1. Row Filtering
- Function: `load_raw_dataframe(parquet_path)`
- Keeps only rows where `LBL_NOK` is not null
- Resets index for deterministic tensor indexing

### B2. Cycle Length Calculation
- Function: `_compute_cycle_length(row)`
- Uses `DXP_TrigClpCls` as reference trigger
- Cycle length = last true/positive index + 1
- Fallback to `DXP_Inj1PrsAct` length if trigger is absent

### B3. Fixed-Length Time-Series Tensor
- Function: `_build_time_series_tensor(df, t_max)`
- For each of 8 channels:
  - convert to float array
  - cast trigger channels to binary float
  - right-pad with zeros or truncate to `t_max`
- Output shape: `(N, 8, T_max)`

### B4. Scalar Feature Matrix
- Function: `_build_scalar_matrix(df)`
- Extracts 5 scalar features
- Includes robust fallback for cavity pressure column naming
- Builds `material_is_PP` from `MET_MaterialName`
- Output shape: `(N, 5)`

---

## C. Normalization Strategy (Leakage-Safe)
File: `preprocess.py`

### C1. Train-Only Fit
`preprocess_probays(...)` accepts `train_indices` and fits stats only on those rows.

### C2. Time-Series Normalization by Material
- Functions:
  - `_fit_time_series_stats_by_material(...)`
  - `_apply_time_series_norm_by_material(...)`
- Separate means/stds for ABS and PP (2 x channels)
- Avoids distortion from material-specific process ranges

### C3. Scalar Normalization
- Functions:
  - `_fit_scalar_stats(...)`
  - `_apply_scalar_norm(...)`
- Standard z-score on training rows only

### C4. Returned Objects
`preprocess_probays` returns:
- `x_ts`: `torch.float32`, `(N, 8, T_max)`
- `x_scalar`: `torch.float32`, `(N, 5)`
- `y`: `torch.float32`, `(N,)`
- `norm_stats`: dict for reproducible re-application

---

## D. Model Implementation
File: `tcn_model.py`

### D1. CausalConv1d
- Computes `left_pad = (kernel_size - 1) * dilation`
- Applies `F.pad(x, (left_pad, 0))`
- Runs `Conv1d(..., padding=0)`
- Prevents future timestep leakage

### D2. TCNBlock
- Sequence:
  - CausalConv1d -> ReLU -> Dropout
  - CausalConv1d -> ReLU -> Dropout
- Residual path:
  - identity if channel counts match
  - 1x1 projection when channel counts differ
- Output: `ReLU(main + residual)`

### D3. TCNDefectClassifier
- Stacked TCN blocks with dilation schedule `[1, 2, 4]`
- Global average pooling over time axis
- Concatenate pooled features with 5 scalar features
- MLP classifier outputs one logit per sample

### D4. FocalLoss
- Uses `binary_cross_entropy_with_logits`
- Adds focal modulation and alpha balancing
- Better behavior on imbalanced defect classes

### D5. LSTM Baseline
- 2-layer LSTM (`hidden=128`, dropout 0.1)
- Global average pooling over sequence
- Same scalar-fusion classifier head pattern

---

## E. Training Orchestration
File: `train.py`

### E1. Reproducibility and Split
- Sets seeds
- Creates stratified held-out test split (20%)
- Remaining train/validation pool used for 5-fold CV

### E2. Dataloaders
- Training uses `WeightedRandomSampler`
- Validation/test uses deterministic non-shuffled loader

### E3. Optimizer and Schedule
- Adam: `lr=1e-3`, `weight_decay=1e-4`
- ReduceLROnPlateau on validation loss
- Gradient clipping with `max_norm=1.0`

### E4. Early Stopping and Thresholding
- Early stopping based on best validation F1
- Threshold chosen by scanning `[0.05, 0.95]` for max F1

### E5. CV to Test Inference
- For each fold:
  - train model
  - save best state + fold normalization stats
- For held-out test:
  - run each fold model with its own stats
  - average fold probabilities (ensemble)

---

## F. Baseline Workflows

### F1. LSTM Baseline
- Same split and CV protocol as TCN
- Same threshold optimization logic
- Same held-out test ensembling strategy

### F2. Logistic Regression Baseline
- Uses scalar features only
- StandardScaler fit on train/val pool only
- Class balancing enabled
- Threshold tuned on train probabilities

---

## G. Evaluation Outputs
Metrics computed from held-out test probabilities:
- AUC-ROC
- AUC-PR
- F1 for positive class (`LBL_NOK = 1`)
- Confusion matrix (`TN, FP, FN, TP`)
- Full classification report

Latest run summary:
- TCN: AUC-ROC 0.9941, AUC-PR 0.9849, F1 0.9375
- LSTM: AUC-ROC 0.9917, AUC-PR 0.9803, F1 0.9123
- Logistic Regression: AUC-ROC 0.8489, AUC-PR 0.7172, F1 0.7042

---

## H. Operational Entry Points
- Main training script: `train.py`
- One-command wrapper with dependency checks: `run_training.py`
- Dependency lock file: `requirements.txt`

Typical run:

```bash
pip install -r requirements.txt
python run_training.py
```

---

## I. Why This Implementation Performs Well
- Causal convolutions enforce physically valid online prediction constraints
- Dilation expands context without exploding parameter count
- Material-aware normalization stabilizes feature scales
- Weighted sampling and focal loss improve minority-class learning
- Threshold tuning aligns decisions with defect-recall/F1 objectives
- Cross-fold ensembling reduces variance and improves generalization
