"""
src/metrics/range_eer.py  --  Range-based EER for spoof localization.

Faithful to Zhang et al., "Range-Based Equal Error Rate for Spoof Localization",
Interspeech 2023 (arXiv:2305.17739), Eqs. (5)-(7).

Idea: unlike point-based EER (count misclassified fixed-resolution segments), range-based
EER measures the *duration* of misclassified regions between the reference and hypothesis,
with NO fixed temporal grid. It is still threshold-free (sweep the threshold to the point
where the two rates are equal).

Definitions (paper Eqs. 5-7), with our convention  higher score = more likely SPOOF:
  overlap(a, b) = max(0, min(a_end, b_end) - max(a_start, b_start))           (Eq. 7)
  At threshold tau, a hypothesis range h is 'predicted spoof' iff score(h) >= tau.
  FPR(tau) = [sum over bonafide-reference x spoof-predicted-hyp of overlap] / D_bonafide
  FNR(tau) = [sum over spoof-reference   x bonafide-predicted-hyp of overlap] / D_spoof
  range-EER = (FPR(tau_hat) + FNR(tau_hat)) / 2   at the tau where |FPR-FNR| is minimised.

Inputs are lists of (start_sec, end_sec, label) reference ranges and
(start_sec, end_sec, score) hypothesis ranges, per clip; we accumulate durations across
all clips (dataset-level EER, the standard population).

NOTE (validation): this is a faithful re-implementation of the paper's equations via a
direct threshold sweep (the paper additionally uses a quantile binary search purely for
speed; the sweep yields the same EER). Before use as the *headline* number it should be
cross-checked against the authors' reference implementation
(github.com/nii-yamagishilab/PartialSpoof), as the project plan requires by week 8.
"""
from typing import List, Tuple
import numpy as np

Range = Tuple[float, float, float]   # (start_sec, end_sec, value)  value = label or score


def overlap(a_start, a_end, b_start, b_end) -> float:
    """Eq. (7): overlapping duration (seconds) of two ranges; 0 if disjoint."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _accumulate(ref_ranges: List[List[Range]],
                hyp_ranges: List[List[Range]]):
    """
    Pre-compute, for every threshold-independent piece, the overlap between each hypothesis
    range and the bonafide/spoof reference duration it covers. Returns, per hypothesis range
    across all clips: (score, overlap_with_spoof_ref, overlap_with_bonafide_ref), plus the
    total spoof/bonafide reference durations D_P, D_N.
    """
    hyp_records = []   # (score, ov_spoof_ref, ov_bona_ref)
    D_P = 0.0          # total spoof reference duration
    D_N = 0.0          # total bonafide reference duration
    for refs, hyps in zip(ref_ranges, hyp_ranges):
        for (rs, re, rl) in refs:
            dur = re - rs
            if rl == 1:  # spoof reference
                D_P += dur
            else:        # bonafide reference
                D_N += dur
        for (hs, he, sc) in hyps:
            ov_sp = 0.0
            ov_bo = 0.0
            for (rs, re, rl) in refs:
                ov = overlap(hs, he, rs, re)
                if ov <= 0:
                    continue
                if rl == 1:
                    ov_sp += ov
                else:
                    ov_bo += ov
            hyp_records.append((sc, ov_sp, ov_bo))
    return hyp_records, D_P, D_N


def _rates_at(hyp_records, D_P, D_N, tau):
    """
    FPR(tau) and FNR(tau) per Eqs. (5)-(6), higher-score-is-spoof convention.
      predicted spoof iff score >= tau.
      FP duration = bonafide-ref duration covered by spoof-predicted hyps
      FN duration = spoof-ref   duration covered by bonafide-predicted hyps
    """
    fp = 0.0  # bonafide ref overlapped by 'spoof' prediction
    fn = 0.0  # spoof ref overlapped by 'bonafide' prediction
    for (sc, ov_sp, ov_bo) in hyp_records:
        if sc >= tau:            # predicted spoof
            fp += ov_bo          #   ... over bonafide reference -> false positive duration
        else:                    # predicted bonafide
            fn += ov_sp          #   ... over spoof reference    -> false negative duration
    fpr = fp / D_N if D_N > 0 else 0.0
    fnr = fn / D_P if D_P > 0 else 0.0
    return fpr, fnr


def range_eer(ref_ranges: List[List[Range]],
              hyp_ranges: List[List[Range]]) -> Tuple[float, float]:
    """
    Range-based EER over all clips. Returns (eer, threshold_at_eer).
    ref_ranges[k] : list of (start, end, label)  for clip k  (label: 1 spoof, 0 bonafide)
    hyp_ranges[k] : list of (start, end, score)   for clip k  (score: higher = more spoof)
    """
    hyp_records, D_P, D_N = _accumulate(ref_ranges, hyp_ranges)
    if D_P == 0 or D_N == 0:
        return 0.5, 0.5          # only one reference class present -> undefined
    if not hyp_records:
        return 0.5, 0.5

    scores = sorted({r[0] for r in hyp_records})
    # candidate thresholds: each distinct score, plus just-below-min and just-above-max
    cand = [scores[0] - 1e-9] + scores + [scores[-1] + 1e-9]
    best = (1.0, cand[0], np.inf)   # (eer, tau, gap)
    for tau in cand:
        fpr, fnr = _rates_at(hyp_records, D_P, D_N, tau)
        gap = abs(fpr - fnr)
        if gap < best[2]:
            best = ((fpr + fnr) / 2.0, tau, gap)
    return float(best[0]), float(best[1])


# --------------------------------------------------------------------------- #
# convenience: convert a per-frame mask/score sequence into ranges
# --------------------------------------------------------------------------- #
def mask_to_ranges(mask: np.ndarray, frame_ms: int = 20) -> List[Range]:
    """Turn a 0/1 frame mask into contiguous reference ranges (start_sec,end_sec,label)."""
    mask = np.asarray(mask).astype(int)
    ranges = []
    if len(mask) == 0:
        return ranges
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            ranges.append((start * frame_ms / 1000.0,
                           i * frame_ms / 1000.0,
                           int(mask[start])))
            start = i
    return ranges


def scores_to_ranges(scores: np.ndarray, frame_ms: int = 20) -> List[Range]:
    """Each frame becomes a uniform hypothesis range carrying its score (uniform case)."""
    scores = np.asarray(scores, dtype=float)
    return [(i * frame_ms / 1000.0, (i + 1) * frame_ms / 1000.0, float(s))
            for i, s in enumerate(scores)]
