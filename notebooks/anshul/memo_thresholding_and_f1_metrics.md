# Technical Memo: Thresholding Dynamics, Metric Taxonomy, and Macro-F1 Behavior

**Date:** 14 November 2026  
**From:** Anshul Rathore (`ee3231024`)  
**To:** Aditya Ranjan Sinha (`ee1230446`)  
**Project:** CompSpoof-TL (B.Tech Thesis, IIT Delhi)  
**Subject:** Clarification on Frame vs. Clip F1, Eval-Set EER Thresholding, and Empirical Sweeps  

---

### 1. Executive Summary & Clarification of Current Metrics

1. **What Was Aditya's Headline 0.674 Macro-F1?**
   * It is the **5-class Utterance-Level (Clip-Level) Macro-F1**, evaluated across: `bonafide_bonafide`, `bonafide_spoof`, `spoof_bonafide`, `spoof_spoof`, and `original`.
   * It operates at the **clip level (9,800 samples)** via max-pooling across the 200 frames per clip:
     $$\text{Clip Decision} = \text{spoof} \iff \max_{t=1}^{200} s(t) \ge T$$
   * It was evaluated at the canonical sigmoid midpoint **$T = 0.50$** (defined in `src/metrics/evaluate.py`), **not** tuned on training data.
2. **What Is the Pure Frame-Level F1?**
   * In `evaluate_suite()`, `frame_f1` evaluates all **1,960,000 frames** independently (1 sample = 1 frame, 20 ms). Every frame carries equal weight.

---

### 2. Investigation: What Happens if We Threshold at Frame-EER?

We ran an experiment on PARAM Pragya (`scores_eval.npz`, 9,800 eval clips) setting the decision threshold to the Frame-EER operating point ($\tau_{\text{sp}} = 0.0147, \tau_{\text{env}} = 0.0251$):

* **Result:** Utterance Macro-F1 drops sharply from **0.6792 $\rightarrow$ 0.4913**.
* **Why does this happen? (Temporal Error Compounding):**
  * At Frame-EER, the false-alarm rate on negative frames is ~18% ($FAR_{\text{frame}} \approx FRR_{\text{frame}}$).
  * A 4-second clip contains 200 frames. The probability that at least one negative frame randomly fires above $\tau \approx 0.015$ is:
    $$P(\text{clip false alarm}) = 1 - (1 - 0.18)^{200} \approx 99.9\%$$
  * Nearly all clean clips are misclassified as `spoof_spoof`, collapsing bonafide precision and pulling down unweighted 5-class Macro-F1.

---

### 3. Empirical Threshold Sweep on the Eval Split (Seed 42)

Sweeping $\tau \in [0.005, 0.900]$ demonstrates where each metric actually peaks:

| Threshold ($\tau$) | Operating Regime | **Utterance Macro-F1** (5-class, clip-level) | **Frame Macro-F1** (4-class, frame-by-frame) | **Speech Frame-F1** (Binary, frame-level) | **Env Frame-F1** (Binary, frame-level) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0.005** | Extreme Low | 0.4139 | 0.4350 | 0.5018 | 0.4358 |
| **0.015** | **Frame-EER Point** | **0.4913** | **0.5025** | 0.5705 | 0.4917 |
| **0.050** | Low | 0.5808 | 0.5638 | 0.6243 | 0.5289 |
| **0.100** | Intermediate | 0.6222 | 0.5867 | 0.6443 | 0.5435 |
| **0.300** | Moderate | 0.6704 | 0.6241 | 0.6694 | 0.5691 |
| **0.500** | **Aditya's Default** | **0.6792** | **0.6399** | **0.6779** | **0.5790** |
| **0.600** | Near-Optimal | 0.6792 | 0.6448 | **0.6786** | **0.5816** |
| **0.700** | **Empirical Peak** | **0.6834** | **0.6461** | 0.6763 | 0.5782 |
| **0.800** | High | 0.6752 | 0.6418 | 0.6705 | 0.5656 |

* **Optimal Utterance Macro-F1:** **0.6836** at $(\tau_{\text{sp}} = 0.675, \tau_{\text{env}} = 0.700)$. Aditya's default ($0.6792$) is within $0.004$ of the theoretical ceiling.
* **Optimal Frame 4-Class Macro-F1:** **0.6461** at $\tau = 0.700$.
* **Optimal Binary Frame-F1:** **0.6789** (Speech, $\tau = 0.590$) and **0.5821** (Env, $\tau = 0.570$).

---

### 4. Recommendations for the Thesis & Paper Draft

1. **Metric Taxonomy**: Clearly distinguish the **Frame-Level Layer** (`range_eer`, `frame_eer`, `frame_f1` over 1.96M frames) from the **Utterance-Level Layer** (5-class Macro-F1 over 9,800 clips used for ESDD2 compatibility and Hypothesis H1).
2. **Reporting Policy**:
   * For **Threshold-Free Localization**: Report Range-EER and Frame-EER (0.190 / 0.223).
   * For **Hard-Decision Metrics**: Keep $T = 0.50$ (or report curves). Explicitly note that frame-level EER operating points cannot be naively max-pooled to clips due to the 200-frame false-alarm accumulation effect.
