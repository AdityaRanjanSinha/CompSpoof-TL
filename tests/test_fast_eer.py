"""The vectorised EER / range-EER must match the original loop versions exactly."""
import numpy as np
from src.metrics.frame_metrics import eer, _eer_reference
from src.metrics.range_eer import (range_eer, _range_eer_reference,
                                   mask_to_ranges, scores_to_ranges)


def test_fast_eer_matches_loop():
    rng = np.random.default_rng(0)
    for trial in range(30):
        n = rng.integers(5, 400)
        lab = rng.integers(0, 2, n)
        sc = rng.random(n) + 0.4 * lab
        if trial % 3 == 0:
            sc = np.round(sc, 1)            # force ties
        assert np.isclose(eer(sc, lab)[0], _eer_reference(sc, lab)[0])


def test_fast_range_eer_matches_loop():
    rng = np.random.default_rng(1)
    for trial in range(15):
        refs, hyps = [], []
        for _ in range(rng.integers(1, 6)):
            m = np.zeros(200, int)
            for _ in range(rng.integers(0, 3)):
                s = rng.integers(0, 180); m[s:s + rng.integers(5, 60)] = 1
            sc = rng.random(200) * 0.6 + 0.4 * m
            if trial % 2:
                sc = np.round(sc, 1)
            refs.append(mask_to_ranges(m)); hyps.append(scores_to_ranges(sc))
        assert np.isclose(range_eer(refs, hyps)[0], _range_eer_reference(refs, hyps)[0])
