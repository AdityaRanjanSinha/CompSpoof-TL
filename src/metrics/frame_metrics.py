"""
src/metrics/frame_metrics.py  --  Frame-level evaluation metrics for CompSpoof-TL.

Localization is scored per component (speech, environment) as a sequence-labelling
problem: at each 20 ms frame the model emits a spoof score in [0,1] and the ground
truth is 0/1 (spoofed or not).

This module provides the FRAME-LEVEL layer (the base of the metric stack):
  - eer(scores, labels)          : Equal Error Rate (threshold-free), + its threshold
  - frame_f1(scores, labels, th) : F1 at a given decision threshold
  - pool_to_resolution(x, k)     : max-pool a 20 ms sequence to a coarser resolution
  - frame_eer_multi_res(...)     : EER at 20/40/160/640 ms, per the plan

Conventions:
  scores : float array, higher = more likely SPOOF (the positive class)
  labels : int/bool array, 1 = spoof (positive), 0 = bonafide
  EER    : lower is better; 0.5 = chance. Reported as a fraction (not %).
"""
from typing import Dict, List, Tuple
import numpy as np


# --------------------------------------------------------------------------- #
# Equal Error Rate
# --------------------------------------------------------------------------- #
def weighted_eer_sweep(scores, w_pos, w_neg, tot_pos=None, tot_neg=None):
    """
    Vectorised threshold sweep shared by frame EER and range EER.
    Each item has a score and a positive (spoof) weight and a negative (bonafide) weight.
    Predict spoof iff score >= t.  FAR(t) = neg weight with score >= t / tot_neg,
    FRR(t) = pos weight with score < t / tot_pos.  Candidates: below-min, every distinct
    score, above-max (ascending); the first minimiser of |FAR-FRR| wins (same tie rule as
    the original loop).  O(N log N) instead of O(N^2).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    w_pos = np.asarray(w_pos, dtype=np.float64).ravel()
    w_neg = np.asarray(w_neg, dtype=np.float64).ravel()
    tot_pos = w_pos.sum() if tot_pos is None else float(tot_pos)
    tot_neg = w_neg.sum() if tot_neg is None else float(tot_neg)
    uniq, inv = np.unique(scores, return_inverse=True)
    pw = np.bincount(inv, weights=w_pos, minlength=len(uniq))
    nw = np.bincount(inv, weights=w_neg, minlength=len(uniq))
    cpw = np.concatenate([[0.0], np.cumsum(pw)])   # pos weight with score < uniq[k]  (k=0..U)
    cnw = np.concatenate([[0.0], np.cumsum(nw)])
    # thresholds: [min-1] -> k=0 ; uniq[k] -> k ; [max+1] -> k=U
    fn = np.concatenate([[0.0], cpw[:-1], [cpw[-1]]])
    fp = np.concatenate([[cnw[-1]], cnw[-1] - cnw[:-1], [0.0]])
    thr = np.concatenate([[uniq[0] - 1], uniq, [uniq[-1] + 1]])
    far = fp / tot_neg if tot_neg > 0 else np.zeros_like(fp)
    frr = fn / tot_pos if tot_pos > 0 else np.zeros_like(fn)
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2.0), float(thr[k])


def eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Equal Error Rate: sweep the threshold; EER is where the false-acceptance rate
    (FAR: bonafide scored as spoof) equals the false-rejection rate (FRR: spoof
    scored as bonafide). Returns (eer, threshold_at_eer).

    Degenerate cases (only one class present) return eer = 0.5 (undefined -> chance).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        return 0.5, 0.5
    return weighted_eer_sweep(scores, labels == 1, labels == 0)


def _eer_reference(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Original O(N^2) loop, kept only to test the fast version against."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    pos = scores[labels == 1]           # spoof scores
    neg = scores[labels == 0]           # bonafide scores
    if len(pos) == 0 or len(neg) == 0:
        return 0.5, 0.5

    # candidate thresholds = every distinct score (pl+/-inf handled by sweep)
    thr = np.unique(np.concatenate([scores, [scores.min() - 1, scores.max() + 1]]))
    thr.sort()
    best_eer, best_th, best_gap = 1.0, thr[0], np.inf
    for t in thr:
        # predict spoof iff score >= t
        far = np.mean(neg >= t)         # bonafide wrongly called spoof
        frr = np.mean(pos < t)          # spoof wrongly called bonafide
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_eer, best_th = gap, (far + frr) / 2.0, t
    return float(best_eer), float(best_th)


# --------------------------------------------------------------------------- #
# F1 at a threshold
# --------------------------------------------------------------------------- #
def frame_f1(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """Binary F1 for the spoof (positive) class at a fixed decision threshold."""
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    pred = (scores >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


# --------------------------------------------------------------------------- #
# multi-resolution pooling (20 ms base -> 40/160/640 ms)
# --------------------------------------------------------------------------- #
def pool_to_resolution(seq: np.ndarray, base_ms: int, target_ms: int, is_score: bool):
    """
    Coarsen a 20 ms sequence to a coarser resolution by MAX-pooling
    (PartialSpoof convention: a coarse block is 'spoof' if ANY fine frame is spoof).
    For labels this is a hard OR; for scores it's the max score in the block.
    Trailing frames that don't fill a full block are pooled as a final short block.
    """
    if target_ms == base_ms:
        return np.asarray(seq)
    k = target_ms // base_ms
    if k < 1 or target_ms % base_ms != 0:
        raise ValueError(f"target_ms {target_ms} must be a multiple of base_ms {base_ms}")
    seq = np.asarray(seq, dtype=np.float64)
    n = len(seq)
    n_blocks = int(np.ceil(n / k))
    out = np.empty(n_blocks)
    for b in range(n_blocks):
        out[b] = seq[b * k:(b + 1) * k].max()
    return out if is_score else (out > 0).astype(int)


# --------------------------------------------------------------------------- #
# EER at multiple resolutions (concatenated across clips)
# --------------------------------------------------------------------------- #
def frame_eer_multi_res(scores_list: List[np.ndarray],
                        labels_list: List[np.ndarray],
                        base_ms: int = 20,
                        resolutions_ms=(20, 40, 160, 640)) -> Dict[int, float]:
    """
    Per-component frame EER at each resolution, computed over ALL clips pooled together
    (scores/labels concatenated across the dataset, then EER once per resolution).
    scores_list / labels_list: one (T,) array per clip.
    """
    out: Dict[int, float] = {}
    for res in resolutions_ms:
        S, L = [], []
        for sc, lb in zip(scores_list, labels_list):
            S.append(pool_to_resolution(sc, base_ms, res, is_score=True))
            L.append(pool_to_resolution(lb, base_ms, res, is_score=False))
        S = np.concatenate(S); L = np.concatenate(L)
        out[res] = eer(S, L)[0]
    return out
