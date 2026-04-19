# ProBayes Defect Prediction with Temporal Convolutional Networks (TCN)

## 1. Project Overview
This project implements a production-oriented multivariate time-series defect prediction system for plastic injection molding cycles from the ProBayes dataset.

Goal:
Predict whether a cycle produces a defective part (`LBL_NOK = 1`) before manual quality inspection.

Task type:
Binary classification on mixed data modalities:
- Time-series process signals per cycle
- Scalar process/context features per cycle

Primary model:
Temporal Convolutional Network (TCN) with causal dilated convolutions.

Baselines:
- LSTM classifier
- Logistic Regression (scalar-only)

## 2. Dataset Context
- Dataset file: `dataset_V2.parquet`
- Total labeled samples used: 564 cycles
- Target: `LBL_NOK` (0 = OK, 1 = defective)

Source domain:
Injection molding experiments (PP and ABS materials), with full-cycle machine sensor traces and quality labels.

## 3. Technical Architecture

### 3.1 Input Modalities
Time-series channels (`in_channels = 8`):
1. `DXP_Inj1PrsAct`
2. `DXP_Inj1PosAct`
3. `TCE_TemperatureMainLine`
4. `TCN_TemperatureMainLine`
5. `DOS_acComp1DosRate`
6. `DXP_TrigInj1` (cast to float)
7. `DXP_TrigHld1` (cast to float)
8. `DXP_TrigCool` (cast to float)

Scalar features (`n_scalar_feats = 5`):
1. `QUA_InjectionPressureMax`
2. `QUA_CavityPressureMax` (fallback to `QUA_CavityPressure1Max` if needed)
3. `QUA_CycleTime`
4. `material_is_PP`
5. `ENV_AirTemperature`

### 3.2 Causal TCN Design
Model class: `TCNDefectClassifier` in `tcn_model.py`.

Building blocks:
- `CausalConv1d`: manual left-only padding via `F.pad((left_pad, 0))`
- `TCNBlock`: two causal conv layers + residual connection
- Exponential dilations across blocks: `[1, 2, 4]`
- Channel progression: `[64, 64, 128]`
- Temporal global average pooling
- Concatenation with scalar features
- MLP head: `Linear(final_dim, 32) -> ReLU -> Dropout -> Linear(32, 1)`
- Output is a single logit (no sigmoid in forward)

Critical causality constraint:
No `padding='same'` usage in `nn.Conv1d`; all temporal padding is manual and left-only.

### 3.3 Receptive Field
Implemented reference formula comment:

`total receptive field = 1 + (kernel_size - 1) * sum(all dilation values across all blocks)`

For `kernel_size = 3` and dilations `[1, 2, 4]`:

$RF = 1 + (3 - 1) * (1 + 2 + 4) = 15$

This ensures controlled history coverage while maintaining causal validity.

### 3.4 Loss for Class Imbalance
Class: `FocalLoss`.

Configuration:
- `alpha = 0.75`
- `gamma = 2.0`

Computation:
- BCE with logits per sample
- Focal weighting by $(1 - p_t)^\gamma$
- Positive/negative balancing via `alpha_t`

## 4. Preprocessing and Data Integrity
Pipeline in `preprocess.py`.

### 4.1 Key Steps
1. Load parquet
2. Filter rows where `LBL_NOK` is non-null
3. Compute cycle length from `DXP_TrigClpCls` (last true index + 1)
4. Set fixed `T_max` from 5th percentile of cycle lengths (train-only context)
5. Right-pad/truncate each signal to `T_max`
6. Stack channels in fixed order `(N, C, T_max)`
7. Build scalar matrix `(N, 5)`
8. Fit normalization on training indices only
9. Apply material-aware time-series normalization (separate PP/ABS stats)
10. Convert outputs to `torch.float32`

### 4.2 Leakage Prevention
- All normalization stats fit only on training fold indices
- Held-out test split isolated before model fitting
- CV folds use train-only statistics for each fold

## 5. Training and Evaluation Protocol
Main pipeline in `train.py`.

### 5.1 Configuration
- Optimizer: Adam (`lr = 1e-3`, `weight_decay = 1e-4`)
- Scheduler: ReduceLROnPlateau (`mode='min'`, `patience=5`, `factor=0.5`)
- Batch size: 32
- Max epochs: 100
- Early stopping: patience 15 (monitoring validation F1)
- Gradient clipping: `clip_grad_norm_(max_norm=1.0)`
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Held-out test: stratified 20% split from full labeled data

### 5.2 Accuracy-Oriented Enhancements
- WeightedRandomSampler for class-imbalanced mini-batches
- Validation-threshold tuning (optimize F1, not fixed 0.5)
- Out-of-fold threshold estimation
- Cross-fold ensemble averaging for test-time probabilities (TCN and LSTM)

## 6. Final Results (Held-out Test)
From the completed training run:

| Model | AUC-ROC | AUC-PR | F1 (NOK=1) |
|---|---:|---:|---:|
| TCN | 0.9941 | 0.9849 | 0.9375 |
| LSTM | 0.9917 | 0.9803 | 0.9123 |
| Logistic Regression | 0.8489 | 0.7172 | 0.7042 |

### 6.1 TCN Confusion Matrix
`[[TN, FP], [FN, TP]] = [[79, 3], [1, 30]]`

Interpretation:
- Very strong minority-class detection
- Low false negatives on defective parts
- Best overall trade-off among tested models

## 7. Repository Structure
- `tcn_model.py`: TCN/LSTM models + focal loss
- `preprocess.py`: preprocessing and normalization
- `train.py`: training, CV, ensembling, evaluation
- `run_training.py`: one-command entrypoint with dependency checks
- `requirements.txt`: reproducible dependency versions
- `_check_cols.py`: dataset schema sanity helper

## 8. Reproducible Execution
Install dependencies:

```bash
pip install -r requirements.txt
```

Run with dependency checks:

```bash
python run_training.py
```

Auto-install missing deps then train:

```bash
python run_training.py --install-missing
```

## 9. Engineering Notes
- CPU training works end-to-end; GPU is auto-detected if available.
- `torch.nn.utils.weight_norm` currently emits a deprecation warning in newer torch versions; behavior remains correct.
- Accuracy is intentionally not emphasized; AUC-PR and F1 for `LBL_NOK=1` are the primary operational metrics.
