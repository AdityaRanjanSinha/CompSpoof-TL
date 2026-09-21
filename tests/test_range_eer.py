"""Unit tests for range-based EER (faithful to Zhang et al. 2023, Eqs. 5-7)."""
import numpy as np
from src.metrics.range_eer import (
    overlap, range_eer, mask_to_ranges, scores_to_ranges,
)
from src.metrics.frame_metrics import eer as point_eer


def test_overlap_eq7():
    assert overlap(0, 1, 0.5, 1.5) == 0.5
    assert overlap(0, 1, 1, 2) == 0.0          # touching -> no overlap
    assert overlap(0, 2, 0.5, 1.0) == 0.5
    assert overlap(0, 1, 2, 3) == 0.0          # disjoint


def test_perfect_localization_zero():
    ref = [[(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 4.0, 0)]]
    hyp = [[(0.0, 1.0, 0.1), (1.0, 2.0, 0.9), (2.0, 4.0, 0.1)]]
    e, _ = range_eer(ref, hyp)
    assert abs(e) < 1e-9


def test_hand_computed_half():
    # spoof [0,1] D_P=1, bona [1,2] D_N=1; at thr .5 FPR=FNR=0.5
    ref = [[(0.0, 1.0, 1), (1.0, 2.0, 0)]]
    hyp = [[(0.0, 0.5, 0.9), (0.5, 1.0, 0.1), (1.0, 1.5, 0.9), (1.5, 2.0, 0.1)]]
    e, _ = range_eer(ref, hyp)
    assert abs(e - 0.5) < 1e-9


def test_hand_computed_quarter():
    ref = [[(0.0, 1.0, 1), (1.0, 2.0, 0)]]
    hyp = [[(0.0, 0.75, 0.9), (0.75, 1.0, 0.1), (1.0, 1.75, 0.1), (1.75, 2.0, 0.9)]]
    e, _ = range_eer(ref, hyp)
    assert abs(e - 0.25) < 1e-9


def test_single_class_returns_chance():
    ref = [[(0.0, 2.0, 1)]]                     # only spoof reference
    hyp = [[(0.0, 2.0, 0.5)]]
    assert range_eer(ref, hyp)[0] == 0.5


def test_mask_to_ranges():
    m = np.array([0, 0, 1, 1, 1, 0])
    assert mask_to_ranges(m, 20) == [(0.0, 0.04, 0), (0.04, 0.10, 1), (0.10, 0.12, 0)]


def test_uniform_ranges_equal_point_eer():
    # On uniform equal-duration ranges, range-EER must equal frame point-EER.
    lab = np.zeros(300, int); lab[40:90] = 1; lab[200:250] = 1
    rng = np.random.default_rng(3)
    sc = lab * 0.5 + rng.random(300) * 0.5
    pe, _ = point_eer(sc, lab)
    re, _ = range_eer([mask_to_ranges(lab, 20)], [scores_to_ranges(sc, 20)])
    assert abs(pe - re) < 1e-6


def test_hard_case_nonzero():
    # a genuinely weak detector -> non-trivial range-EER
    rng = np.random.default_rng(11)
    refs, hyps = [], []
    for _ in range(20):
        lab = np.zeros(200, int)
        s = rng.integers(10, 120); d = rng.integers(20, 60); lab[s:s + d] = 1
        sc = lab * 0.4 + rng.random(200) * 0.6
        refs.append(mask_to_ranges(lab, 20))
        hyps.append(scores_to_ranges(sc, 20))
    e, _ = range_eer(refs, hyps)
    assert 0.05 < e < 0.5
