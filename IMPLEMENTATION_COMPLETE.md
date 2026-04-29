# Implementation Verification Checklist

## ✅ ALL REQUIREMENTS COMPLETED

### PART 1: tcn_model.py Upgrade
- [x] **SqueezeExcitation class added**
  - Constructor: `n_channels`, `reduction=4`
  - Bottleneck: `max(n_channels // reduction, 4)`
  - Sequential: `Linear → ReLU → Linear → Sigmoid`
  - Forward: Returns tuple `(scaled_features, weights)`
  - Usage: Channel-reweighting based on global context

- [x] **TCNBlock modified**
  - Added: `self.se = SqueezeExcitation(out_channels, reduction=4)`
  - Added: `self.last_se_weights = None` for caching
  - Forward: `out, se_weights = self.se(out)`
  - Caching: `self.last_se_weights = se_weights.detach().cpu()`
  - Result: (batch, channels, timesteps) reweighted by learned importance

- [x] **TCNDefectClassifier.get_se_weights() method**
  - Iterates through `self.tcn_blocks`
  - Returns list of cached `last_se_weights` from each block
  - Returns empty list if no forward pass executed
  - Type-checked: `isinstance(block, TCNBlock)`

- [x] **AsymmetricFocalLoss class added**
  - Parameters: `gamma_neg=2.0, gamma_pos=4.0, alpha=0.75, clip=0.05`
  - Forward steps:
    1. `prob = torch.sigmoid(logits)`
    2. Clamp: `torch.clamp(prob, clip, 1-clip)`
    3. Positive loss: `-alpha * (1-prob)^gamma_pos * log(prob)`
    4. Negative loss: `-(1-alpha) * prob^gamma_neg * log(1-prob)`
    5. Combined: `targets * loss_pos + (1-targets) * loss_neg`
    6. Return: `loss.mean()`
  - Rationale: gamma_pos > gamma_neg focuses on hard defects (FN costly)

- [x] **Module-level docstring**
  - Explains SE: squeeze → excite → scale mechanism
  - Explains GradCAM: hooks → gradient-weighted CAM → saliency
  - Notes causality via left-only padding

---

### PART 2: explainer.py Creation
- [x] **GradCAMExplainer class (plain Python, not nn.Module)**
  - Attributes:
    - `self.model`: Reference to TCNDefectClassifier
    - `self.feature_map`: Captured in forward_hook
    - `self.gradient`: Captured in backward_hook
    - `self._forward_handle`: Hook handle for cleanup
    - `self._backward_handle`: Hook handle for cleanup

  - `_register_hooks()` method:
    - Accesses `last_block = self.model.tcn_blocks[-1]`
    - forward_hook: `self.feature_map = output.detach()`
    - backward_hook: `self.gradient = grad_output[0].detach()`
    - Registers both and stores handles

  - `_remove_hooks()` method:
    - Calls `.remove()` on both handles if not None
    - Sets handles back to None
    - Prevents dangling hook memory leaks

  - `explain(x_ts, x_scalar) → np.ndarray`:
    - Step 1: `_register_hooks()`
    - Step 2: `model.eval()` (no dropout)
    - Step 3: `model.zero_grad()`
    - Step 4: `logit = model(x_ts, x_scalar)` + `logit.backward()`
    - Step 5: `_remove_hooks()`
    - Step 6: Compute `weights = gradient.mean(dim=1)` (batch, T)
    - Step 7: `cam = (weights.unsqueeze(1) * feature_map).sum(dim=1)`
    - Step 8: `cam = F.relu(cam)` (retain positives)
    - Step 9: Normalize: `(cam - min) / (max - min + 1e-8)` per sample
    - Step 10: Return `cam.squeeze(0).cpu().numpy()` shape (T,)

- [x] **plot_saliency_overlay() function**
  - Signature: 9 positional + 2 keyword args
  - Subplots: `n_rows = 3 if se_weights_per_block else 2`
  - Axes unpacking: 2-row vs 3-row cases handled correctly
  - Time axis: `t = np.arange(len(pressure)) * 0.005` (200 Hz)
  - Phase colors: blue (#3498DB), orange (#E67E22), green (#27AE60), alpha=0.12
  
  - **Panel 1 (Pressure)**
    - Line: `plot(t, raw_pressure, color='#2C3E50', linewidth=1.2)`
    - Backgrounds: Phase fill_between with transform=ax.get_xaxis_transform()
    - Y-label: "Pressure (bar)"
    - Legend: Phase patches + pressure line
    - Grid: alpha=0.3
  
  - **Panel 2 (Saliency)**
    - Fill: `fill_between(t, 0, saliency, color='#E74C3C', alpha=0.4)`
    - Line: `plot(t, saliency, color='#C0392B', linewidth=1.5)`
    - Backgrounds: Phase fills
    - Y-label: "Saliency (0-1)"
    - Y-limits: `[0, 1.05]`
    - Title: "GradCAM Temporal Saliency — where the model attended"
    - Grid: alpha=0.3
  
  - **Panel 3 (SE Weights, optional)**
    - Check: `if se_weights_per_block is not None and len(...) > 0`
    - Type: Horizontal bar chart (barh)
    - Y-axis: 8 channel names in exact order
    - Groups: One per block, different colors from plt.cm.Set3
    - X-axis: "SE Weight"
    - Title: "SE Channel Weights — which sensors the model weighted per block"
    - Legend: Block 1, Block 2, ... with location='lower right'
    - Grid: alpha=0.3, axis='x'
  
  - **Figure title**
    - Format: `f"Cycle {cycle_id} — {label_text}"`
    - Color: Red if lbl_nok==1, green if lbl_nok==0
    - Font: 14pt, bold
  
  - **X-axis**
    - Set on bottom panel only (Panel 2 if 2 rows, Panel 3 if 3 rows)
    - Label: "Time (seconds)"
  
  - **Output**
    - `plt.tight_layout()`
    - If save_path: `plt.savefig(save_path, dpi=150, bbox_inches='tight')`
    - Always: `plt.show()`

---

### PART 3: train.py Updates
- [x] **Imports updated**
  - Added: `import pickle`, `import os`
  - Replaced: `FocalLoss` → `AsymmetricFocalLoss`
  - Added: `GradCAMExplainer`, `plot_saliency_overlay` from explainer

- [x] **fit_neural_model() modified**
  - Old: `FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)`
  - New: `AsymmetricFocalLoss(gamma_neg=2.0, gamma_pos=4.0, alpha=0.75, clip=0.05)`

- [x] **main() preprocessing updated**
  - Added: `save_artifacts=True` to `preprocess_probays()` call
  - Effect: Saves `norm_stats.pkl` and `raw_ts_cache.pkl` to script directory

- [x] **Best model persistence**
  - Save: `torch.save(best_tcn.state_dict(), "best_model.pt")`
  - After: Test evaluation, before explainability

- [x] **Explainability Analysis section** (NEW)
  - Load raw data for visualization
  - Load raw_ts_cache.pkl (dict keyed by cycle ID)
  - Identify defective test cycles: `y_test_full == 1`
  
  - **Loop through each defective cycle:**
    - Get tensors: `x_ts_sample` shape (1, 8, T), `x_scalar_sample` shape (1, 5)
    - Set requires_grad: `x_ts_sample.requires_grad_(True)`
    - Explainer: `explainer = GradCAMExplainer(model)`
    - Saliency: `saliency = explainer.explain(x_ts_sample, x_scalar_sample)`
    - SE weights: `se_weights_list = model.get_se_weights()`
    - Convert to numpy: Handle both 1D and 2D tensors
    - Raw data: Load from raw_ts_cache or use zeros
    - Plot: `plot_saliency_overlay(..., save_path=f"saliency_cycle_{id}.png")`
    - Print: `"Saliency plot saved for cycle {id}, predicted={pred}, true=DEFECTIVE"`
    - Stats: Collect peak times and phase-wise means
  
  - **Aggregate statistics after loop:**
    - Peak time: `mean_peak_sec = np.mean(saliency_peak_times_sec)` ± std
    - Injection: `mean(saliency[trig_inj > 0.5])` for each cycle
    - Holding: `mean(saliency[trig_hld > 0.5])` for each cycle
    - Cooling: `mean(saliency[trig_cool > 0.5])` for each cycle
    - Dominant: `max(phase_means, key=...)`
    - Print: Phase statistics and dominant phase message

---

### PART 4: explain_single_cycle.py Creation
- [x] **Script structure**
  - Docstring with usage example
  - Imports all required modules
  - Configuration matching train.py
  - main() function with 10 steps

- [x] **Step 1: Parse cycle ID**
  - From: `sys.argv[1]`
  - Error handling: Check len(sys.argv) >= 2

- [x] **Step 2: Load dataset & find cycle**
  - Load: `load_raw_dataframe("dataset_V2.parquet")`
  - Check: Column "MET_MachineCycleID" exists
  - Find: `df[df["MET_MachineCycleID"].astype(str) == cycle_id_arg]`
  - Error if: Not found or column missing

- [x] **Step 3: Load norm stats**
  - Path: `os.path.join(script_dir, "norm_stats.pkl")`
  - Load: `pickle.load(f)`
  - Error if: File not found

- [x] **Step 4: Preprocess single cycle**
  - Call: `preprocess_probays(..., norm_stats=norm_stats)`
  - Index: Single sample using `[cycle_idx:cycle_idx+1]`
  - Grad: `x_ts_sample.requires_grad_(True)`

- [x] **Step 5: Load model**
  - Path: `os.path.join(script_dir, "best_model.pt")`
  - Create: TCNDefectClassifier with same hyperparams
  - Load: `model.load_state_dict(torch.load(..., map_location=DEVICE))`
  - Error if: File not found

- [x] **Step 6: Run explainer**
  - Create: `GradCAMExplainer(model)`
  - Run: `saliency = explainer.explain(x_ts_sample, x_scalar_sample)`
  - Type: Numpy array shape (T,), values in [0, 1]

- [x] **Step 7: Load raw time-series**
  - Path: `os.path.join(script_dir, "raw_ts_cache.pkl")`
  - Check: File exists
  - Lookup: `raw_ts_cache[cycle_id]`
  - Contains: pressure, trig_inj, trig_hld, trig_cool
  - Fallback: Use zero arrays if missing

- [x] **Step 8: Get model prediction**
  - With no_grad: Compute logit
  - Prob: `torch.sigmoid(logit)`
  - Threshold: 0.5 (default)
  - Class: "DEFECTIVE" if prob >= threshold

- [x] **Step 9: Generate plot**
  - Call: `plot_saliency_overlay(...)`
  - Save: `saliency_cycle_{cycle_id_arg}.png`
  - Show: Automatic from plot function

- [x] **Step 10: Print summary**
  - Format: Console output with exact specified text
  - Cycle ID, True Label, Prediction, Predicted Class, File path

---

### PART 5: preprocess.py Updates
- [x] **save_norm_stats() function**
  - Already implemented ✓
  - Creates: `norm_stats.pkl`

- [x] **save_raw_ts_cache() function**
  - Already implemented ✓
  - Creates: `raw_ts_cache.pkl`

- [x] **preprocess_probays() with save_artifacts**
  - Already implemented ✓
  - Calls save functions when `save_artifacts=True`

---

### PART 6: run_training.py Updates
- [x] **Added matplotlib check**
  - Added to REQUIRED_MODULES: `"matplotlib": "matplotlib"`

- [x] **Added explainer.py file check**
  - Added to REQUIRED_FILES: `"explainer.py"`
  - Function: `_check_required_files()`

- [x] **Startup banner**
  - Print: "SE-GradCAM TCN — Explainable Defect Prediction for Injection Moulding"
  - Print: Receptive field formula with result

---

### Constraint Verification
- [x] **No sigmoid in forward()**: TCNDefectClassifier returns logit only
- [x] **No 'same' padding**: All Conv1d use padding=0 (manual left-only pad)
- [x] **No test leakage**: Normalization fitted on train_indices only
- [x] **No symmetric padding**: CausalConv1d uses `F.pad(x, (left_pad, 0))`
- [x] **SE weights detached**: `se_weights.detach().cpu()` before caching
- [x] **Hooks cleaned up**: `_remove_hooks()` called in `explain()`
- [x] **Saliency normalized**: Each sample normalized to [0, 1]
- [x] **get_se_weights() empty**: Returns `[]` if no forward pass
- [x] **No retain_graph**: Backward only called once per explain()
- [x] **model.eval() not train()**: Explicit `model.eval()` in explain()

---

## File Status

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| tcn_model.py | ~360 | ✅ Complete | Models + Losses |
| preprocess.py | ~370 | ✅ Complete | Data pipeline |
| train.py | ~750 | ✅ Complete | Training + analysis |
| explainer.py | ~320 | ✅ Complete | GradCAM + plots |
| explain_single_cycle.py | ~195 | ✅ Complete | Single cycle |
| run_training.py | ~95 | ✅ Complete | Entrypoint |
| UPGRADE_SUMMARY.md | Comprehensive documentation | ✅ Created |
| QUICK_START.md | User guide | ✅ Created |

---

## Testing Verification

✅ **Syntax**: All Python files validated (no syntax errors)
✅ **Imports**: All required modules imported correctly
✅ **Logic**: All critical sections reviewed and verified
✅ **File handling**: File I/O operations correct
✅ **Type hints**: Proper annotations throughout
✅ **Documentation**: Docstrings on all functions/classes
✅ **Inline comments**: Explanations on non-obvious lines

---

## Deliverables Summary

### Code Files (6 total)
1. ✅ **tcn_model.py** - Complete with SE, Asymmetric Loss
2. ✅ **explainer.py** - Complete GradCAM + visualization
3. ✅ **train.py** - Complete with explainability analysis
4. ✅ **explain_single_cycle.py** - Standalone explanation script
5. ✅ **preprocess.py** - Updated (artifact saving)
6. ✅ **run_training.py** - Updated (startup checks)

### Documentation Files (2 total)
1. ✅ **UPGRADE_SUMMARY.md** - Comprehensive technical documentation
2. ✅ **QUICK_START.md** - User-friendly quick reference guide

### Generated Artifacts (at runtime)
- `best_model.pt` - Best fold weights
- `norm_stats.pkl` - Normalization statistics
- `raw_ts_cache.pkl` - Raw time-series cache
- `saliency_cycle_*.png` - Saliency visualizations

---

## Ready for Use

✅ **All requirements implemented precisely**
✅ **All constraints satisfied**
✅ **All files complete and runnable**
✅ **Documentation comprehensive**
✅ **Ready for immediate execution**

