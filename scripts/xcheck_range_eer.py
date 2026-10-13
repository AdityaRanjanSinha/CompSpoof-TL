#!/usr/bin/env python
# Usage: pip install pyannote.metrics pyannote.core ; python scripts/xcheck_range_eer.py .
"""
Cross-check of our range-based EER (src/metrics/range_eer.py) against the authors'
reference logic (PartialSpoof/metric/RangeEER.py, Zhang et al. 2023):
  - hypothesis = contiguous 'spoof' segments where the frame is predicted spoof at threshold th
    (their scoali2seg), evaluated with pyannote DetectionCostFunction over the whole clip (uem)
  - FPR = false-alarm duration / negative total, FNR = miss duration / positive total
  - EER found by their percentile bisection (prec 1e-5)
"""
import sys
import numpy as np
from pyannote.core import Annotation, Segment
import pyannote.metrics.detection as det
from scipy.stats import percentileofscore

sys.path.insert(0, sys.argv[1])
from src.metrics.range_eer import range_eer, _accumulate, _rates_at, mask_to_ranges, scores_to_ranges

FR = 0.02


def ref_annotation(mask):
    a = Annotation()
    for s, e, l in mask_to_ranges(mask):
        if l == 1:
            a[Segment(s, e)] = "spoof"
    return a


def hyp_annotation(spoof_pred):                 # their scoali2seg (run-length -> segments)
    a = Annotation()
    i, n = 0, len(spoof_pred)
    while i < n:
        if spoof_pred[i]:
            j = i
            while j < n and spoof_pred[j]:
                j += 1
            a[Segment(i * FR, j * FR)] = "spoof"
            i = j
        else:
            i += 1
    return a


def ref_rates(masks, scores, th):
    """Reference FPR/FNR at threshold th (predicted spoof iff spoof score >= th)."""
    d = det.DetectionCostFunction()
    for m, s in zip(masks, scores):
        d(reference=ref_annotation(m), hypothesis=hyp_annotation(s >= th), uem=Segment(0, len(m) * FR))
    acc = d.accumulated_
    fpr = acc["false alarm"] / acc["negative class total"] if acc["negative class total"] else 0.0
    fnr = acc["miss"] / acc["positive class total"] if acc["positive class total"] else 0.0
    return fpr, fnr


def ref_bisection_eer(masks, scores, prec=1e-5):
    allsc = np.sort(np.concatenate(scores))
    lo, hi = allsc[0], allsc[-1]
    lo_p, hi_p = 0.0, 100.0
    mid_p = 50.0; mid = np.percentile(allsc, mid_p)
    f_lo, n_lo = ref_rates(masks, scores, lo)
    f_mid, n_mid = ref_rates(masks, scores, mid)
    res = [(mid, f_mid, n_mid)]
    it = 0
    while lo < hi and abs(f_mid - n_mid) > prec and it < 60:
        if (f_lo - n_lo) * (f_mid - n_mid) < 0:
            hi, hi_p = mid, mid_p
        else:
            lo, lo_p, f_lo, n_lo = mid, mid_p, f_mid, n_mid
        mid_p = (lo_p + hi_p) / 2; mid = np.percentile(allsc, mid_p)
        f_mid, n_mid = ref_rates(masks, scores, mid)
        res.append((mid, f_mid, n_mid)); it += 1
    best = min(res, key=lambda r: abs(r[1] - r[2]))
    return (best[1] + best[2]) / 2


def main():
    rng = np.random.default_rng(0)
    worst_rate, rows = 0.0, []
    for trial in range(8):
        masks, scores = [], []
        for _ in range(60):
            m = np.zeros(200, int)
            for _ in range(rng.integers(0, 3)):
                st = rng.integers(0, 170); m[st:st + rng.integers(10, 90)] = 1
            s = rng.random(200) * 0.7 + 0.35 * m + 0.1 * rng.standard_normal(200)
            masks.append(m); scores.append(s)
        refs = [mask_to_ranges(m) for m in masks]; hyps = [scores_to_ranges(s) for s in scores]
        rec, DP, DN = _accumulate(refs, hyps)
        # (a) FPR/FNR at fixed thresholds must agree exactly
        for th in np.quantile(np.concatenate(scores), [0.1, 0.3, 0.5, 0.7, 0.9]):
            a = _rates_at(rec, DP, DN, th); b = ref_rates(masks, scores, th)
            worst_rate = max(worst_rate, abs(a[0] - b[0]), abs(a[1] - b[1]))
        ours = range_eer(refs, hyps)[0]
        theirs = ref_bisection_eer(masks, scores)
        rows.append((ours, theirs))
        print(f"trial {trial}: ours {ours:.5f}  reference(bisection) {theirs:.5f}  diff {abs(ours - theirs):.2e}")
    print(f"max |FPR/FNR difference| at fixed thresholds: {worst_rate:.2e}")
    print(f"max |EER difference|: {max(abs(a - b) for a, b in rows):.2e}")


if __name__ == "__main__":
    main()
