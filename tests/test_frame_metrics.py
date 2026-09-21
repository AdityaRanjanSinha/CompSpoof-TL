"""Unit tests for frame-level metrics (EER, F1, multi-resolution pooling)."""
import numpy as np
from src.metrics.frame_metrics import (
    eer, frame_f1, pool_to_resolution, frame_eer_multi_res,
)


def test_eer_perfect_separation():
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    e, _ = eer(scores, labels)
    assert abs(e - 0.0) < 1e-9


def test_eer_random_is_chance():
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    labels = rng.integers(0, 2, 2000)          # scores independent of labels
    e, _ = eer(scores, labels)
    assert 0.42 < e < 0.58


def test_eer_known_value():
    # 2/10 spoof look bonafide, 2/10 bonafide look spoof -> EER provably 0.2
    spoof = np.array([0.9] * 8 + [0.1] * 2)
    bona = np.array([0.1] * 8 + [0.9] * 2)
    scores = np.concatenate([spoof, bona])
    labels = np.array([1] * 10 + [0] * 10)
    e, _ = eer(scores, labels)
    assert abs(e - 0.2) < 1e-9


def test_eer_single_class_returns_chance():
    assert eer(np.array([0.3, 0.7]), np.array([1, 1]))[0] == 0.5
    assert eer(np.array([0.3, 0.7]), np.array([0, 0]))[0] == 0.5


def test_f1_perfect_and_miss():
    s = np.array([0.9, 0.9, 0.1, 0.1]); l = np.array([1, 1, 0, 0])
    assert abs(frame_f1(s, l, 0.5) - 1.0) < 1e-9
    assert frame_f1(np.array([0.1, 0.1]), np.array([1, 1]), 0.5) == 0.0


def test_pool_labels_or():
    lab = np.array([0, 0, 1, 0, 0, 0, 0, 1])
    assert pool_to_resolution(lab, 20, 40, is_score=False).tolist() == [0, 1, 0, 1]


def test_pool_scores_max():
    sc = np.array([0.1, 0.3, 0.2, 0.8, 0.4, 0.1, 0.9, 0.2])
    assert np.allclose(pool_to_resolution(sc, 20, 40, is_score=True),
                       [0.3, 0.8, 0.4, 0.9])


def test_pool_requires_multiple():
    import pytest
    with pytest.raises(ValueError):
        pool_to_resolution(np.zeros(10), 20, 30, is_score=False)   # 30 not multiple of 20


def test_multi_res_hard_case_nonzero_and_monotone_ish():
    # a genuinely imperfect detector so EERs are non-trivial
    rng = np.random.default_rng(7)
    scores_list, labels_list = [], []
    for _ in range(30):
        lab = np.zeros(200, int)
        start = rng.integers(10, 120); dur = rng.integers(20, 60)
        lab[start:start + dur] = 1
        # weak detector: signal + substantial noise so it's far from perfect
        sc = lab * 0.5 + rng.random(200) * 0.6
        scores_list.append(sc); labels_list.append(lab)
    res = frame_eer_multi_res(scores_list, labels_list)
    # should be a real, non-zero EER at the finest resolution
    assert res[20] > 0.05
    # coarser resolutions are usually easier (lower EER) since errors get pooled away;
    # allow slack but check 640ms is not worse than 20ms by a lot
    assert res[640] <= res[20] + 0.05
