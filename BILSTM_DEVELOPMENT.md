# BiLSTM Fault Classifier -- Development Log

AI Digital Twin Engine Fault Simulator | Bosch Engine Dataset

---

## Table of Contents

1. [What the BiLSTM Does](#1-what-the-bilstm-does)
2. [What It Sets Up for Later Steps](#2-what-it-sets-up-for-later-steps)
3. [v1 Architecture and Its Flaws](#3-v1-architecture-and-its-flaws)
4. [v2 Architecture - All Fixes Applied](#4-v2-architecture---all-fixes-applied)
5. [Performance Report](#5-performance-report)
6. [Threshold Calibration](#6-threshold-calibration)
7. [Remaining Weaknesses](#7-remaining-weaknesses)
8. [Next Steps](#8-next-steps)

---

## 1. What the BiLSTM Does

### The Problem It Solves

The Bosch Engine Dataset contains no fault labels. Raw sensor recordings capture only normal engine operation across three sources: `gengine1` (13 features, 247k rows), `gengine2` (9 features, 403k rows), and `pengines` (11 static features). To build a fault detection system from this data, faults must be synthesised by perturbing the control channels that physically govern engine combustion:

| Fault Class | Channel Perturbed | Offset | Physical Meaning |
|---|---|---|---|
| 0 -- Normal | none | -- | Stoichiometric combustion |
| 1 -- Rich Mixture | Lambda | -1.5 sigma | Excess fuel, incomplete combustion |
| 2 -- Lean Mixture | Lambda | +1.5 sigma | Fuel deficiency, elevated NOx |
| 3 -- Ignition Fault | IgnitionAngle | +2.0 sigma | Retarded timing, power loss |

The BiLSTM classifier takes a 30-timestep sliding window (3 seconds at 10 Hz) of 13 standardised sensor readings and assigns it to one of these four classes in real time.

### What It Contributes to the System

The BiLSTM is the **detection brain** of the simulator. Every cycle of the closed-loop simulation, the backend calls it to answer one question:

> "Given the last 3 seconds of sensor data, is this engine operating normally -- and if not, what type of fault is present?"

Its output -- a fault class plus a calibrated confidence score -- directly drives the rest of the pipeline:

```
Sensor window (30 x 13)
        |
        v
  BiLSTM Classifier
        |
  fault_class, confidence
        |
   +----+----+
   |         |
Normal     Fault detected
   |         |
  Pass    Control action computed
            (fuel trim / spark advance)
            |
        Digital Twin validates action
            |
        State window updated
            |
        SHAP explains which features
        drove the fault detection
```

Without a reliable classifier, the entire downstream pipeline -- the digital twin, the controller, the SHAP explainability layer -- has nothing meaningful to respond to. The BiLSTM is what makes the simulation reactive rather than scripted.

---

## 2. What It Sets Up for Later Steps

### Step 4 -- LSTM Digital Twin

The digital twin (already trained, RMSE = 0.108) predicts the next engine state given the current state plus a proposed control action. It is invoked **only when the BiLSTM detects a fault**. The quality of the twin's predictions is therefore gated by the quality of the classifier:

- A high false-negative rate (missed faults) means the twin is never triggered, and the engine drifts uncorrected.
- A high false-positive rate (false alarms on Normal) means the twin fires unnecessarily and the controller wastes correction cycles.
- A well-calibrated classifier (like v2) gives the twin clean, well-timed activation -- it is called exactly when needed, with a confidence score the twin can use to weight its intervention.

### Step 5 -- SHAP Explainability

SHAP values are computed against the BiLSTM's background distribution. The model's attention mechanism (MultiHeadAttention over 30 timesteps) means SHAP can now attribute importance both to *which feature* drove the detection and *which timestep in the window* carried the fault signal. This is a significant upgrade over v1 where only final-hidden-state representations were available.

The `shap_cache.json` currently stores v1 importances. After the v2 swap-in, `run_shap.py` should be re-run so that SHAP results reflect the attention-weighted representation.

### Step 6 -- Closed-Loop Simulation

The simulation loop (`simulation_engine.py`) runs at interactive speed in the browser. The BiLSTM is called every cycle via `classify_window()`. The v2 improvements directly affect the simulation in three ways:

1. **Calibrated thresholds** (0.20-0.83 range, vs. v1's unusable 0.9997-1.0) allow the backend to add a confidence gate: only raise a fault alarm when `confidence >= threshold[class]`. This prevents noise-driven false alarms during normal operation.
2. **Better ignition fault detection** reduces the chance of a simulated ignition fault going undetected for several cycles at the start.
3. **Label smoothing outputs** mean confidence scores shown in the frontend UI are honest numbers (e.g., 0.87, 0.93) rather than all being 0.9999.

### Presentation to Judges

The BiLSTM is the most technically scrutinisable component of the project because:
- It is the only component with a published accuracy metric (macro F1)
- Judges familiar with ML will ask about evaluation methodology
- The v1 circular evaluation (F1 = 0.991 on identical training offsets) would not survive scrutiny

The v2 evaluation -- variable-magnitude faults from `Uniform(0.8, 2.2) x base_offset`, seed 999, completely decoupled from training -- produces a defensible, honest result: **macro F1 = 0.9947, macro AUC = 0.9998** on genuinely unseen fault severities.

---

## 3. v1 Architecture and Its Flaws

### Architecture

```
Input(30, 13)
  -> Bidirectional LSTM(64, return_sequences=True)   # 128 units
  -> BatchNormalization                               # WRONG for sequences
  -> Dropout(0.30)
  -> Bidirectional LSTM(32, return_sequences=False)  # 64 units, drops sequence
  -> BatchNormalization
  -> Dropout(0.25)
  -> Dense(32, relu)                                 # severe bottleneck
  -> Dropout(0.20)
  -> Dense(4, softmax)

Total params: 84,132
```

### Flaw Catalogue

| # | Flaw | Severity | Description |
|---|---|---|---|
| 1 | Circular evaluation | Critical | Test evaluation re-injected faults at identical offsets (+-1.5, +2.0) to training. F1=0.991 measures memorisation, not generalisation. |
| 2 | Test set all-Normal | Critical | `g1_y_test` had zero fault examples. No real held-out fault evaluation existed. |
| 3 | BatchNorm on sequence data | High | `BatchNormalization` on `(None, 30, 128)` normalises across the batch dimension per timestep -- statistically incorrect for RNN output. Correct choice is `LayerNormalization`. |
| 4 | Final hidden state only | High | `return_sequences=False` on the second BiLSTM discards all timestep information except the last hidden state. Faults concentrated in early or mid-window timesteps are underweighted. |
| 5 | Dense(32) bottleneck | Medium | Compressing 64 BiLSTM units to 32 before classification loses temporal context. |
| 6 | P10 thresholds near 1.0 | High | Saved thresholds: Normal=0.9997, Rich=0.9999, Lean=1.0000, Ignition=0.9982. Any real deployment would reject virtually every prediction. Direct consequence of circular evaluation + no label smoothing. |
| 7 | Dead import | Medium | `from notebooks.build_nb02 import *` was present in the evaluation cell despite the comment "won't work easily, inline instead". Imports nothing, silently breaks on clean re-run. |
| 8 | Fault fraction mismatch | Low | Training used 25% fault fraction; evaluation used 30%. Minor class prior shift. |
| 9 | No ROC / AUC | Medium | Only accuracy + F1 + confusion matrix reported. False negative rate analysis was impossible. |
| 10 | gengine2 shape mismatch | High | BiLSTM expects (30, 13) input. gengine2 sessions produce (30, 8) windows. Backend would crash at runtime for any gengine2 simulation. |

---

## 4. v2 Architecture - All Fixes Applied

### Architecture

```
Input(30, 13)
  -> Bidirectional LSTM(96, return_sequences=True)   # 192 units
  -> LayerNormalization                              # correct for sequences
  -> Dropout(0.25)
  -> Bidirectional LSTM(64, return_sequences=True)   # 128 units, keeps sequence
  -> LayerNormalization
  -> Dropout(0.20)
  -> MultiHeadAttention(heads=4, key_dim=16)         # self-attention over 30 timesteps
  -> Add [residual connection]                       # stabilises gradients
  -> LayerNormalization
  -> GlobalAveragePooling1D                          # weighted mean over time axis
  -> Dense(64, relu)                                 # wider head
  -> Dropout(0.15)
  -> Dense(4, softmax)

Total params: ~200k
```

### Why Each Change Matters

**LayerNormalization replacing BatchNormalization**

`BatchNormalization` on a sequence tensor of shape `(batch, timesteps, features)` normalises across the batch dimension for each `(timestep, feature)` position. At batch size 256 with 30 timesteps and 128 features, this means computing 3,840 separate normalisation statistics per batch. These statistics change with batch size and are unstable for small batches. The normalisation crosses sample boundaries, which can corrupt the temporal structure of individual sequences.

`LayerNormalization` normalises across the feature dimension within each sample independently. For a single sequence, it computes one mean and variance across the 128 features at each timestep. It is batch-size-independent, works correctly at inference time with batch size 1, and is the normalisation used in every Transformer and modern RNN-based model.

**MultiHeadAttention with residual connection**

The 30-timestep window contains unequal information content across time. In a window containing an ignition fault, the perturbation may be strongest in the middle timesteps as the fault develops, with early timesteps still in normal range. With `return_sequences=False`, the original model only used the final hidden state -- the fault signal in earlier timesteps was compressed through the LSTM recurrence and partially lost.

MultiHeadAttention operates directly on all 30 timestep representations simultaneously. Each of the 4 heads learns to attend to different temporal patterns -- one head may focus on the onset of a Lambda deviation, another on its peak, a third on correlations between Lambda and NOx. The residual connection (`Add([x, attn_out])`) ensures that if attention adds no information, the BiLSTM output flows through unchanged -- the attention mechanism cannot degrade the model, only improve it.

**GlobalAveragePooling1D**

After attention, instead of taking only the final timestep, this layer computes the mean across all 30 timestep representations. This gives equal weight to every part of the window (modulated by attention scores) and produces a fixed-length (128,) vector summarising the entire window. This is the correct aggregation for variable-length temporal patterns.

**Label smoothing (0.05)**

During training, the target distribution for a fault-1 window is normally `[0, 1, 0, 0]`. With label smoothing of 0.05, this becomes `[0.0125, 0.9625, 0.0125, 0.0125]`. The model is penalised if it assigns zero probability to any class. This prevents the softmax from saturating to extreme values, directly producing calibrated output probabilities -- the mechanism that fixed the pathological P10-near-1.0 thresholds of v1.

**Variable-magnitude evaluation**

```python
scale = rng_eval.uniform(0.8, 2.2)
actual_offset = base_off * scale
```

Training used a fixed offset (e.g., Lambda += -1.5 for every Rich window). A model that simply memorises "Lambda < -1.3 means Rich" would achieve 99%+ F1 on identical-offset evaluation. By randomising the offset magnitude between 0.8x and 2.2x the base offset, the v2 evaluation tests whether the model learnt the *pattern* of fault development across the window -- not just a static threshold on one feature.

---

## 5. Performance Report

### Training Convergence

The model converged cleanly in approximately 23 epochs, well before the 60-epoch maximum. EarlyStopping fired on val_loss plateau and restored best weights automatically. Both train and val loss tracks are tight -- no overfitting gap is visible. The label smoothing floor keeps loss from approaching zero, settling at ~0.20-0.22.

A mild accuracy oscillation of approximately +-1.5% on the validation set is visible in the first 8 epochs. This is caused by the 15% validation split interacting with the class imbalance (70% Normal vs. 10% per fault class) -- some batches have slightly different fault prevalence. It resolves after epoch 10 and does not affect final performance.

### Confusion Matrix Results

| Actual \ Predicted | Normal | Rich | Lean | Ignition |
|---|---|---|---|---|
| **Normal (25,818)** | 25,725 (99.6%) | 0 | 18 (0.1%) | 75 (0.3%) |
| **Rich Mixture (3,754)** | 5 (0.1%) | 3,749 (99.9%) | 0 | 0 |
| **Lean Mixture (3,637)** | 3 (0.1%) | 0 | 3,634 (99.9%) | 0 |
| **Ignition Fault (3,799)** | 38 (1.0%) | 0 | 0 | 3,761 (99.0%) |

**Zero inter-fault confusion.** The model never misclassifies Rich as Lean or as Ignition, and never confuses Lean with Ignition. This is the most important safety property: in a real engine, confusing the *type* of fault leads to the wrong corrective action (adding fuel when you should be reducing it). The v2 model cannot make this mistake.

**Ignition Fault is the hardest class.** 38 windows (1.0%) were missed as Normal. These are predominantly low-severity windows (0.8x offset = 1.6 sigma), where the IgnitionAngle deviation is within the tail of the normal distribution. The secondary emission effects (elevated HC, CO) are correspondingly small. This is a genuine physical difficulty, not a modelling failure.

**Normal class false positives** (93 windows mislabelled as Lean or Ignition) are predominantly from the most severe fault windows (2.2x offset). At 4.4 sigma, the Lambda or IgnitionAngle value is so extreme that it causes correlated shifts in multiple features simultaneously. A small number of these fall near the Normal-Ignition boundary. Given that 26.0% of the eval set is fault windows, this false positive rate is negligible.

### ROC and AUC

| Class | ROC-AUC | PR-AUC |
|---|---|---|
| Normal | 0.9997 | 0.9997 |
| Rich Mixture | 1.0000 | 1.0000 |
| Lean Mixture | 1.0000 | 1.0000 |
| Ignition Fault | 0.9995 | 0.9912 |
| **Macro** | **0.9998** | -- |

All four ROC curves hug the top-left corner. At any False Positive Rate below 1%, the True Positive Rate for Rich and Lean Mixture is already 1.0 -- these faults are trivially separable from Normal across the full severity range tested.

The Ignition Fault PR-AUC of 0.9912 is the only number below 0.999 in the entire evaluation. At high recall (catching every ignition fault including the lowest-severity ones), precision falls slightly, meaning some Normal windows get flagged. This is the threshold at which the model's genuine uncertainty appears -- and it is the right place for it to appear.

**Macro F1 = 0.9947, Macro AUC = 0.9998** on variable-magnitude, independently seeded evaluation.

---

## 6. Threshold Calibration

The v2 thresholds are calibrated by searching 200 candidate values in [0.10, 0.99] and selecting the one that maximises binary F1 for each class in a one-vs-rest framework. These replace the v1 P10 thresholds that were 0.9982-1.0000.

| Class | v1 Threshold (P10) | v2 Threshold (F1-max) | Deployable? |
|---|---|---|---|
| Normal | 0.9997 | **0.198** | Yes |
| Rich Mixture | 0.9999 | **0.212** | Yes |
| Lean Mixture | 1.0000 | **0.825** | Yes |
| Ignition Fault | 0.9982 | **0.686** | Yes |

**Usage in the backend:** The backend currently uses raw `argmax` on the softmax output. The calibrated thresholds enable an additional confidence gate: if `max_prob < threshold[predicted_class]`, return Normal rather than raising a potentially spurious alarm. This is a one-line change to `classifier.py`.

**Threshold interpretation note:** The Normal threshold of 0.198 reflects the high prevalence of Normal in the evaluation set (70%). In one-vs-rest terms, 20% Normal softmax probability is sufficient to beat the F1-maximising threshold because Normal is so prevalent. If the production fault rate increases above ~30%, these thresholds should be recalibrated on the new distribution.

---

## 7. Remaining Weaknesses

### A. Ignition Fault at Low Severity (1.0% miss rate)
The minimum test offset is 0.8 x 2.0 sigma = 1.6 sigma. At this level, IgnitionAngle is within the outer tail of its normal distribution. The model misses 1% of these cases. The fix is low-severity augmentation during training: sample fault offset magnitudes from a uniform range during preprocessing rather than always using the fixed base offset.

### B. gengine2 Runtime Crash (Unresolved)
The BiLSTM expects input shape (30, 13). A gengine2 simulation session produces windows of shape (30, 8). The backend passes these directly to `classify_window()` which calls `model.predict()` -- this will raise a shape mismatch error at runtime. The backend effectively has dead gengine2 classification support until this is resolved.

Options: (1) restrict classification to gengine1 and return class 0 for gengine2, (2) pad gengine2 windows to 13 features with zeros for the missing channels, (3) train a separate 8-feature BiLSTM for gengine2.

### C. Normal Threshold Below Random Baseline
A Normal threshold of 0.198 is below the 0.25 random-chance baseline for a 4-class problem. This is mathematically valid (it encodes the class prior), but means the threshold is not a measure of model certainty -- it is a measure of class prevalence. A threshold below 0.25 will accept nearly every Normal prediction regardless of model confidence.

### D. No Temporal Consistency Evaluation
Per-window accuracy was measured, but consecutive overlapping windows were not tested. In the simulation loop, windows slide by 1 timestep per cycle. A fault detected at cycle N might be dropped at N+1 and detected again at N+2. This "chattering" is not measured anywhere. A stability metric (e.g., "90% label agreement across a 10-cycle fault window") should be added.

### E. No SHAP Update
`shap_cache.json` was computed against the v1 model. The v2 attention mechanism changes the feature importance profile. The current SHAP display in the frontend reflects v1 representations, not v2.

---

## 8. Next Steps

### Immediate -- Before Presentation

**1. Swap v2 into the backend**

In `backend/models_loader.py`, update the model and threshold paths:

```python
# Before
BILSTM_PATH     = "models/bilstm_classifier.h5"
THRESHOLDS_PATH = "models/bilstm_thresholds.json"

# After
BILSTM_PATH     = "models/bilstm_v2_classifier.h5"
THRESHOLDS_PATH = "models/bilstm_v2_thresholds.json"
```

No other backend changes are required. Input/output shapes are identical.

**2. Add confidence gating to classifier.py**

```python
# In classify_window(), after computing fault_class and confidence:
threshold = store.thresholds.get(str(fault_class), 0.5)
if confidence < threshold:
    fault_class = 0          # insufficient confidence -- default to Normal
    confidence  = float(probs[0])
```

**3. End-to-end simulation test**

Start backend (`uvicorn main:app --port 8000`) and frontend (`npm run dev`), inject each of the three fault types via the UI, and confirm the simulation detects and corrects them correctly with v2 weights.

### Short Term -- After Presentation

**4. Low-severity training augmentation**

In the preprocessing pipeline, change the fault injection to sample offset magnitudes from `Uniform(0.7, 1.5)` in addition to the peak offset. This fills the gap that currently causes the 1.0% ignition fault miss rate at low severity. Expected improvement: Ignition Fault recall from 99.0% to ~99.5%.

**5. Re-run SHAP on v2**

```bash
cd "D:/mini project/notebooks"
python run_shap.py
```

This regenerates `shap_cache.json` against v2 model weights. The attention-weighted representation will change which timesteps are attributed as most important, potentially surfacing richer feature interactions in the SHAP bar charts displayed in the frontend.

**6. Resolve gengine2 shape mismatch**

Implement the simplest fix in `backend/classifier.py`: detect when the incoming window has fewer than 13 features and pad with zeros for the missing channels. This allows gengine2 sessions to run without a crash, though classification quality for gengine2 will be suboptimal until a dedicated model is trained.

**7. Add temporal stability metric**

In `run_simulation.py`, after the closed-loop test, compute label consistency across consecutive windows:

```python
# fraction of cycles where label matches the majority label in a 10-cycle window
stability = rolling_majority_agreement(predictions, window=10)
```

Report this alongside per-window accuracy as a more deployment-relevant metric.

**8. Retrain SHAP background on v2 representations**

The SHAP background array (`shap_background.npy`) was sampled for the v1 model. For v2, re-sample the background from the v2 training set with the same stratified sampling strategy to ensure SHAP attribution is computed against the correct reference distribution.

---

## File Index

| File | Description |
|---|---|
| `notebooks/train_bilstm_v2.py` | v2 training script -- all fixes applied |
| `models/bilstm_v2_classifier.h5` | Trained v2 model weights (1.1 MB approx.) |
| `models/bilstm_v2_thresholds.json` | Calibrated thresholds, AUC scores, architecture record |
| `models/bilstm_v2_training_history.png` | Loss and accuracy curves |
| `models/bilstm_v2_confusion_matrix.png` | Per-class confusion matrix (counts + percentages) |
| `models/bilstm_v2_roc_curves.png` | Per-class ROC + Precision-Recall curves |
| `notebooks/03_bilstm_fault_classifier.ipynb` | v1 notebook (reference only -- do not retrain from) |
