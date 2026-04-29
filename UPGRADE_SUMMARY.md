# SE-GradCAM TCN Implementation Summary

## Overview
Successfully implemented **SE-GradCAM TCN** — a Squeeze-and-Excitation Temporal Convolutional Network with Gradient-weighted Class Activation Mapping for explainable injection molding defect prediction.

## Architectural Contributions

### 1. **Squeeze-and-Excitation (SE) Channel Attention**
- **Location**: Inside each TCNBlock
- **Mechanism**:
  - SQUEEZE: Global average pooling over time axis → per-channel descriptor
  - EXCITE: Two FC layers with ReLU bottleneck + Sigmoid → per-channel weights in [0,1]
  - SCALE: Multiply each channel's time-series by learned weight
- **Benefit**: Model explicitly learns which sensors are most diagnostic for defect prediction

### 2. **GradCAM Temporal Saliency**
- **Hooks**: Registered on final TCNBlock (forward and backward)
- **Computation**: Gradient-weighted feature map combination → per-timestep saliency curve
- **Causality**: Valid because CausalConv1d uses left-only padding (feature at t only sees t≤t)
- **Output**: Normalized [0,1] saliency array showing where model attended

### 3. **Asymmetric Focal Loss**
- **Motivation**: In manufacturing, false negatives (missed defects) are more costly than false positives
- **Implementation**: 
  - `gamma_pos = 4.0`: Aggressively down-weight easy positives → focus on hard defects
  - `gamma_neg = 2.0`: Standard focal weighting for OK parts
  - `alpha = 0.75`: Positive class weighting
  - `clip = 0.05`: Probability clipping for numerical stability

## File Changes

### Modified Files

#### **tcn_model.py**
**Changes:**
- Added `SqueezeExcitation` class (lines ~56-72)
  - Takes `n_channels` and `reduction=4`
  - Returns tuple: (reweighted_features, channel_weights)
  
- Modified `TCNBlock.__init__()` (line ~96)
  - Added: `self.se = SqueezeExcitation(out_channels, reduction=4)`
  - Added: `self.last_se_weights = None`
  
- Modified `TCNBlock.forward()` (lines ~105-109)
  - Applied SE attention: `out, se_weights = self.se(out)`
  - Cached weights (detached, on CPU): `self.last_se_weights = se_weights.detach().cpu()`
  
- Added `TCNDefectClassifier.get_se_weights()` method (lines ~160-169)
  - Returns list of cached weights, one per block
  - Returns empty list if no forward pass yet
  
- Added `AsymmetricFocalLoss` class (lines ~220-251)
  - Separate gamma values for positive/negative samples
  - Probability clipping for stability
  - Combines asymmetric focal weights with alpha balancing

**Module-level docstring**: Updated to explain SE and GradCAM contributions

---

#### **train.py**
**Changes:**
1. **Imports** (lines 1-32):
   - Added: `pickle`, `os`
   - Replaced: `FocalLoss` → `AsymmetricFocalLoss`
   - Added: `GradCAMExplainer`, `plot_saliency_overlay` from explainer

2. **Modified `fit_neural_model()`** (line ~233):
   - Replaced: `FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)`
   - With: `AsymmetricFocalLoss(gamma_neg=2.0, gamma_pos=4.0, alpha=0.75, clip=0.05)`

3. **Modified `main()` preprocessing** (line ~476):
   - Added: `save_artifacts=True` to `preprocess_probays()` call
   - Saves `norm_stats.pkl` and `raw_ts_cache.pkl` for later use

4. **Added model persistence** (after test evaluation, line ~505):
   - Saves best TCN fold model to `best_model.pt`

5. **New: Explainability Analysis Section** (lines ~520-700):
   - Loads raw data for visualization
   - Loads raw_ts_cache.pkl (pressure + trigger arrays)
   - Identifies all defective test cycles
   - For each defective cycle:
     - Runs GradCAMExplainer.explain() → saliency
     - Calls model.get_se_weights() → channel weights
     - Generates saliency plot with save_path=f"saliency_cycle_{id}.png"
     - Prints: "Saliency plot saved for cycle {id}, predicted={pred}, true=DEFECTIVE"
   
   - Aggregates statistics:
     - Mean saliency peak time (seconds) ± std
     - Mean saliency per phase (Injection/Holding/Cooling)
     - Identifies dominant phase

---

### New Files

#### **explainer.py**
**GradCAMExplainer class:**
- `__init__(model)`: Store model, initialize hooks to None
- `_register_hooks()`: Hook into final TCNBlock
  - `forward_hook`: Captures feature_map
  - `backward_hook`: Captures gradient
- `_remove_hooks()`: Clean up hooks (prevents memory leaks)
- `explain(x_ts, x_scalar) → np.ndarray`:
  - Registers hooks
  - Forward pass → captures feature_map
  - Backward pass → captures gradient
  - Removes hooks
  - Computes: `weights = gradient.mean(dim=1)` (per-timestep importance)
  - Computes: `cam = (weights * feature_map).sum(dim=1)` (spatial importance)
  - Applies ReLU (retain positive contributions)
  - Normalizes to [0, 1]
  - Returns numpy array shape (T,)

**plot_saliency_overlay() function:**
- **Arguments:**
  - `raw_pressure`: Unscaled DXP_Inj1PrsAct (bar)
  - `saliency`: GradCAM output [0-1]
  - `trig_inj, trig_hld, trig_cool`: Phase trigger arrays
  - `lbl_nok`: True label (0=OK, 1=DEFECTIVE)
  - `cycle_id`: Cycle identifier string
  - `se_weights_per_block`: Optional list of SE weights per block
  - `save_path`: Optional output file path
  
- **Panels** (sharex=True):
  1. **Pressure (always)**
     - Plot: raw_pressure with phase backgrounds
     - Phases: Injection (#3498DB, α=0.12), Holding (#E67E22, α=0.12), Cooling (#27AE60, α=0.12)
     - Legend: Phase patches + pressure line
     - Y-axis: Pressure (bar)
  
  2. **Saliency (always)**
     - Plot: fill_between + line of saliency with phase backgrounds
     - Y-axis: Saliency (0-1), limits [0, 1.05]
     - Title: "GradCAM Temporal Saliency — where the model attended"
  
  3. **SE Weights (optional, if se_weights_per_block provided)**
     - Type: Grouped horizontal bar chart
     - X-axis: SE weight values
     - Y-axis: 8 channel names (DXP Pressure, DXP Position, TCE Temp, TCN Temp, DOS Rate, Trig Injection, Trig Holding, Trig Cooling)
     - Groups: One per TCN block with different colors
     - Title: "SE Channel Weights — which sensors the model weighted per block"
  
- **Figure title**: "Cycle {id} — {DEFECTIVE (red) | OK (green)}"
- **X-axis**: Time (seconds) = index × 0.005 (200 Hz sampling rate)
- **Output**: Saves to save_path (dpi=150, bbox_inches='tight'), always calls plt.show()

---

#### **explain_single_cycle.py**
**Standalone script for single-cycle explanation**

**Usage:**
```bash
python explain_single_cycle.py <MachineCycleID>
```

**Steps:**
1. Parse cycle ID from command line
2. Load dataset, locate matching row
3. Load norm_stats.pkl (if missing, error with helpful message)
4. Preprocess single cycle using saved statistics
5. Load best_model.pt (if missing, error with helpful message)
6. Reconstruct TCNDefectClassifier with same hyperparameters
7. Run GradCAMExplainer.explain() → saliency
8. Load raw_ts_cache.pkl for pressure/trigger arrays
9. Call plot_saliency_overlay() with save_path=f"saliency_cycle_{id}.png"
10. Print summary:
    ```
    Cycle ID: {id}
    True Label: {DEFECTIVE | OK}
    Model Prediction: {prob:.4f}
    Predicted Class: {DEFECTIVE | OK}
    Saliency plot saved to: {path}
    ```

---

### Updated Files (No code changes, already complete)

#### **preprocess.py**
- `save_norm_stats(norm_stats, path=None)`: Already implemented
  - Saves to `norm_stats.pkl` (default: script directory)
  - Contains: `t_max`, `time_series_cols`, `scalar_cols`, `ts_mean`, `ts_std`, `scalar_mean`, `scalar_std`, `ts_means_by_material`, `ts_stds_by_material`, `scalar_means`, `scalar_stds`

- `save_raw_ts_cache(cache, path=None)`: Already implemented
  - Saves to `raw_ts_cache.pkl` (default: script directory)
  - Dictionary keyed by MET_MachineCycleID, each entry contains:
    - `DXP_Inj1PrsAct`: Unscaled pressure array
    - `DXP_TrigInj1`: Injection phase boolean array
    - `DXP_TrigHld1`: Holding phase boolean array
    - `DXP_TrigCool`: Cooling phase boolean array

- `preprocess_probays(..., save_artifacts=True)`: Already implemented
  - Calls save_norm_stats() and save_raw_ts_cache() if save_artifacts=True
  - Must be called with train_indices only (no test leakage)

---

#### **run_training.py**
**Changes:**
1. Added `"matplotlib": "matplotlib"` to REQUIRED_MODULES
2. Added `REQUIRED_FILES = ["tcn_model.py", "preprocess.py", "train.py", "explainer.py"]`
3. Added `_check_required_files()` function
4. Enhanced startup message with SE-GradCAM TCN banner
5. Print receptive field derivation: `"1 + (3-1) x (1+2+4) = 15 timesteps"`

---

## Key Implementation Details

### Memory Management
- **SE weights**: Detached with `.detach().cpu()` → prevents gradient accumulation
- **Hooks**: Always removed in `_remove_hooks()` → prevents dangling references
- **Gradient tape**: Feature map and gradient are detached immediately after backward()

### Causality
- **CausalConv1d** uses left-only padding: `F.pad(x, (self.left_pad, 0))`
- **GradCAM validity**: Feature at time t only encodes t ≤ t
- **No symmetric padding**: Never uses `padding='same'`

### Numerical Stability
- **Saliency normalization**: Clamp min/max to prevent division by zero: `(cam - min) / (max - min + 1e-8)`
- **Probability clipping**: Clamp prob to [0.05, 0.95] before log in AsymmetricFocalLoss
- **No sigmoid in forward()**: TCNDefectClassifier returns logit only

### Constraints Satisfied
✓ No sigmoid in TCNDefectClassifier.forward()  
✓ No 'same' padding in Conv1d  
✓ No test-data leakage in normalization fitting  
✓ No symmetric padding in CausalConv1d  
✓ SE weights detached before storage  
✓ Hooks removed after explain() to prevent memory leaks  
✓ Saliency always normalized to [0, 1]  
✓ get_se_weights() returns empty list if no forward pass yet  
✓ model.train() NOT called inside explain() method  
✓ Gradient retained only as long as needed  

---

## Expected Output

### During Training
```
SE-GradCAM TCN — Explainable Defect Prediction for Injection Moulding
Receptive field = 1 + (3-1) x (1+2+4) = 15 timesteps

[Training proceeds with 5-fold CV...]

Saved best TCN model to best_model.pt
Saved norm_stats.pkl and raw_ts_cache.pkl to [script_dir]

EXPLAINABILITY ANALYSIS — GradCAM Temporal Saliency

Saliency plot saved for cycle CYCLE_001, predicted defective=DEFECTIVE, true label=DEFECTIVE
Saliency plot saved for cycle CYCLE_002, predicted defective=OK, true label=DEFECTIVE
...

Mean saliency peak time: 0.4235 ± 0.1562 seconds
Mean saliency during injection phase: 0.523
Mean saliency during holding phase: 0.618
Mean saliency during cooling phase: 0.285
The model primarily attended to the Holding phase when predicting defects.
```

### Single Cycle Explanation
```bash
$ python explain_single_cycle.py "CYCLE_001"

Loading dataset...
Found cycle at index 45
Loaded normalization statistics from .../norm_stats.pkl
Loaded model from .../best_model.pt
Generating GradCAM saliency for cycle CYCLE_001...

======================================================================
EXPLANATION SUMMARY
======================================================================
Cycle ID: CYCLE_001
True Label: DEFECTIVE
Model Prediction: 0.8723
Predicted Class: DEFECTIVE
Saliency plot saved to: saliency_cycle_CYCLE_001.png
======================================================================
```

---

## Files Generated

After running train.py:
- `best_model.pt`: Trained model weights (best fold)
- `norm_stats.pkl`: Normalization statistics (t_max, means, stds by material)
- `raw_ts_cache.pkl`: Raw time-series cache (pressure + triggers by cycle ID)
- `saliency_cycle_*.png`: One per defective test cycle (multi-panel visualization)

---

## Testing Checklist

- [x] All imports resolve correctly
- [x] tcn_model.py: SE module integrated into TCNBlock
- [x] tcn_model.py: TCNDefectClassifier.get_se_weights() works
- [x] tcn_model.py: AsymmetricFocalLoss implemented with correct gamma values
- [x] explainer.py: GradCAMExplainer hooks registered and removed
- [x] explainer.py: Saliency normalized to [0, 1]
- [x] explainer.py: plot_saliency_overlay handles 2 or 3 panels correctly
- [x] train.py: AsymmetricFocalLoss instantiation with correct parameters
- [x] train.py: preprocess_probays called with save_artifacts=True
- [x] train.py: best_model.pt saved
- [x] train.py: Explainability analysis loop processes all defective cycles
- [x] explain_single_cycle.py: Standalone script works independently
- [x] run_training.py: matplotlib check added
- [x] run_training.py: explainer.py file check added

---

## Performance Notes

Original model on held-out test (as provided):
- AUC-ROC: 0.9941
- AUC-PR: 0.9849
- F1: 0.9375
- Confusion matrix: TN=79, FP=3, FN=1, TP=30

Expected after upgrade:
- Performance should remain similar or improve (asymmetric focal loss focuses on hard cases)
- Explainability output will identify which sensors drove each prediction
- Saliency curves will show the timing of defect-causing events

---

## References

**Code Comments**: All non-obvious lines have inline comments explaining what and why  
**Docstrings**: All functions and classes have comprehensive docstrings  
**Architecture**: Module-level docstring in tcn_model.py explains SE and GradCAM  

