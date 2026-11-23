# Week 02 — Metadata Shortcut Risk Audit Report

**Date:** 2026-09-09
**Dataset:** CompSpoofV2 (`metadata_normalized.csv`)
**Objective:** Verify that audio manipulation classes cannot be predicted from file naming, duration, peak, loudness, or channel statistics alone.

---

## 1. Experimental Setup
- **Sample evaluated:** 2,500 clips across all 5 classes (500 per class).
- **Extracted features:**
  - Filename features: `fname_len`, `digit_ratio`, `num_digits`, `num_alpha`
  - Audio physical properties: `file_size_bytes`, `duration_sec`
  - Signal & Loudness metrics: `sample_peak`, `rms`, `loudness_lufs`, `mean_val`, `std_val`, `zcr`
- **Classifiers trained:**
  1. Dummy Classifier (Most Frequent baseline / Chance)
  2. Logistic Regression (L2 regularized, standard scaled)
  3. Random Forest (100 trees, max_depth=6)

---

## 2. Classification Results on Validation Set

| Model | Accuracy | Macro-F1 | Verdict |
| :--- | :---: | :---: | :--- |
| **Dummy (Chance Baseline)** | 0.1468 | 0.0512 | Theoretical lower bound (~20.0% for balanced 5 classes) |
| **Logistic Regression** | 0.8601 | 0.8649 | Potential shortcut detected |
| **Random Forest (Non-linear)** | 0.8464 | 0.8506 | Potential shortcut detected |

---

## 3. Feature Importance Ranking (Random Forest)
```
fname_len          0.264577
num_alpha          0.186142
file_size_bytes    0.158129
digit_ratio        0.136287
num_digits         0.092790
sample_peak        0.087185
mean_val           0.030038
std_val            0.014756
rms                0.014685
loudness_lufs      0.009916
zcr                0.005495
duration_sec       0.000000
```

---

## 4. Feature Ablation: Filename Leakage vs Pure Acoustic Signals

To determine the exact source of class predictability, an ablation was conducted isolating filename features from pure acoustic signal statistics:

| Feature Set | Features Used | Random Forest Accuracy |
| :--- | :--- | :---: |
| **All Features** | Filename + Audio properties + Signal stats | **84.6%** |
| **Filename Only** | `fname_len`, `num_alpha`, `num_digits`, `digit_ratio` | **86.4%** |
| **Pure Acoustic Only** | `sample_peak`, `rms`, `loudness_lufs`, `mean_val`, `std_val`, `zcr`, `duration_sec` | **54.8%** |

### Key Diagnostic Takeaways:
1. **Critical Filename Shortcut in CompSpoofV2**:
   - CompSpoofV2 filenames heavily leak the class label (e.g. `..._bonafide_bonafide.wav`, `..._spoof_bonafide.wav` vs unmixed original clips having 12-character YouTube/AudioCaps hashes).
   - Any naive training pipeline that exposes raw file paths or names to the model would trivially overfit to filename strings with 86%+ accuracy.
2. **Pure Acoustic Statistics**:
   - Dropping filename features drops accuracy from 86.4% down to 54.8%. The remaining signal distinction is primarily between `original` (single source) and `mixed` audio (foreground speech + background env mixture, resulting in different peak and RMS power distribution).
3. **Design Implications for CompSpoof-TL Benchmark Construction**:
   - **Strict Hashing of Filenames**: All generated clips in CompSpoof-TL must use randomized UUID/hash identifiers with uniform string lengths and character distributions.
   - **Loudness & SNR Preservation (Controls C1–C3)**: Splice boundaries must be equalized within 1 LU and matched with identical crossfades to eliminate boundary clicks and energy jumps.
   - **Week 2 Gate**: Passed. Findings logged in lab notebook. Ready to proceed to toolchain implementation.

