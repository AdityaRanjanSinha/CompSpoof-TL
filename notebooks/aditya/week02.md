# Week 02 — Acoustic Component Shortcut Audit (follow-up)

**Author:** Aditya
**Date:** 2026-09-20
**Track:** S1 (benchmark / evaluation) — cross-review + follow-up on the Week-2 metadata shortcut audit
**Script:** `scripts/audit_acoustic_component_shortcut.py`
**Data:** CompSpoofV2 source clips (`metadata_normalized.csv`), train+val, mixed classes only

---

## 1. Motivation — the gap this closes

The initial Week-2 shortcut audit (`audit_metadata_shortcuts.py`) reported ~55% accuracy from
**acoustic-only** features on the full 5-class task and concluded the leak was "primarily
original-vs-mixed" (i.e. benign for our component-level task). **That conclusion was assumed,
not tested.**

A simple sanity check shows it is suspect. If the classifier only separated `original` from the
four mixed classes and was at chance among the mixed ones, expected accuracy would be roughly:

    P(original)·1 + P(mixed)·(1/4) ≈ 0.20 + 0.80·0.25 = 0.40

But the audit got **0.55**. The ~15-point excess implies acoustic features also distinguish the
**mixed** classes from one another — which, if true, is a real, task-relevant shortcut, not a
benign one. This follow-up tests that directly.

## 2. Method

- Restrict to **mixed clips only** (drop `original`) — removes the benign original-vs-mixed
  distinction that inflated the earlier number.
- **Acoustic features only** (filename features excluded — the model never sees filenames):
  `duration_sec, sample_peak, rms, loudness_lufs (ITU-R BS.1770), mean_val, std_val, zcr`.
  (Same extraction as the original audit, for comparability.)
- Balanced sample: 800 clips/class × 4 mixed classes = **3,200 clips**.
- Classifiers: Random Forest + Logistic Regression, **5-fold stratified CV**, vs a
  majority-class baseline. Three targeted tests:
  - **A** — 4 mixed classes (chance 0.25)
  - **B** — SPEECH: spoof vs bonafide among mixed (chance 0.50)
  - **C** — ENVIRONMENT: spoof vs bonafide among mixed (chance 0.50)

## 3. Results

| Test | Task | Chance | Best acc (5-CV) | Margin | Verdict |
|:---|:---|:---:|:---:|:---:|:---|
| A | 4 mixed classes | 0.250 | **0.491** (RF) | +0.241 | LEAK |
| B | Speech spoof vs bonafide | 0.500 | **0.661** (RF) | +0.161 | LEAK |
| C | Environment spoof vs bonafide | 0.500 | **0.725** (RF) | +0.225 | LEAK |

(LogReg: A 0.341, B 0.619, C 0.528 — same direction, RF stronger, so the leak is partly
non-linear.)

## 4. Interpretation

- **The "original-vs-mixed only" assumption is refuted.** With `original` fully removed, the
  mixed classes are still separable well above chance on acoustic features alone. So the
  earlier 55% was **not** benign — a substantial part of it was real component-level leakage.
- **Both components leak, environment worst.** Acoustic stats predict spoofed vs bonafide
  **speech at 66%** and **environment at 72.5%**. The environment being the worst is consistent
  with the rest of the project (baseline env EER ≈ 0.43; EnvSDD's unseen-generator findings) —
  generated environmental audio carries a distinctive energy/spectral signature.
- **Root cause (hypothesis):** spoofed source clips (from TTS / audio generators such as
  audioLDM, MeanAudio, etc.) have systematically different loudness / energy / spectral
  statistics than bonafide sources. Naive splicing would inherit this as a giveaway.

## 5. Corrected safety conclusion

**The CompSpoofV2 sources carry a strong acoustic component-level shortcut (speech 66%,
environment 72.5%), not merely an original-vs-mixed one.** The source data is therefore **NOT
"safe" as-is** for constructing a localization benchmark without controls.

## 6. Implications for CompSpoof-TL construction

1. **Loudness / energy equalization (Control C3) is mandatory, not optional.** Spliced segments
   must be loudness-matched (≤1 LU) and energy-normalized so that "generated-audio energy
   signature" cannot stand in for real spoofing evidence.
2. **This audit becomes the verification tool for the C4 / GNG-2 gate.** After construction,
   re-run this exact test on the *built* CompSpoof-TL benchmark; Tests B and C must fall to
   ~0.50 (and A to ~0.25). If they do not, equalization is insufficient and construction is
   revised before any model work.
3. **Open risk — spectral leakage may survive loudness matching.** 66–72% is large; if the
   discriminative signal is spectral (not just level), loudness matching alone may not remove
   it. To be checked during construction; may require spectral matching or acceptance/reporting
   as a known limitation.

## 7. Status

- Follow-up audit complete; finding logged.
- **Action for construction:** treat C3 (loudness/energy equalization) as high-priority; wire
  this audit in as the C4 verification check.
- **Communicated to partner** (owns construction toolchain) so the equalization requirement is
  built in from the start.