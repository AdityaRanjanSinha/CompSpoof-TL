# Range-based EER: cross-check against the authors' reference implementation

**Date:** 2026-10-13 · **Plan item:** week 7–8 "range-based EER verified against reference behaviour"

## What was compared
- **Ours:** `src/metrics/range_eer.py` (Zhang et al. 2023, Eqs. 5–7; vectorised threshold sweep).
- **Reference:** `metric/RangeEER.py` in github.com/nii-yamagishilab/PartialSpoof (authors' code).
  At a threshold, frames predicted spoof are merged into contiguous segments (`scoali2seg`),
  scored with `pyannote.metrics.detection.DetectionCostFunction` over the whole clip
  (FPR = false-alarm duration / negative total, FNR = miss duration / positive total), and the
  EER threshold is found by percentile bisection (precision 1e-5).
- The reference logic was re-implemented faithfully with the same pyannote calls
  (`scripts/xcheck_range_eer.py`), because the original script is tied to their file formats.

## Result
8 random trials (60 clips × 200 frames each, 0–2 spoof regions per clip, noisy scores):

| Check | Max difference |
|---|---|
| FPR and FNR at 5 fixed thresholds per trial | 7e-14 |
| Final range-based EER (ours vs reference bisection) | 6e-14 |

The two agree to floating-point precision. **Our range-EER implementation is verified.**

## Note for reporting
With one score per 20 ms frame and reference boundaries on the 20 ms grid, range-based EER
equals the 20 ms frame EER (merging frames into segments does not change false-alarm or miss
durations). This is a property of the metric, now confirmed with the authors' own logic.
