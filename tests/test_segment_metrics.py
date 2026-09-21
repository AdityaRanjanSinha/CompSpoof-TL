"""Unit tests for event-based F1, joint clip-correct rate, and utterance Macro-F1."""
import numpy as np
from src.metrics.segment_metrics import (
    mask_to_events, event_f1, clip_correct_rate,
    frames_to_utterance_label, macro_f1,
)


def test_mask_to_events():
    m = np.array([0, 1, 1, 0, 0, 1, 0])
    assert mask_to_events(m, 20) == [(0.02, 0.06), (0.10, 0.12)]


def test_event_f1_perfect():
    ref = [np.array([0, 1, 1, 0, 0, 1, 1, 0])]
    assert abs(event_f1(ref, ref, 20, 0.05) - 1.0) < 1e-9


def test_event_f1_collar():
    ref = [np.array([0, 1, 1, 1, 0])]   # onset 0.02
    hyp = [np.array([0, 0, 1, 1, 1])]   # onset 0.04
    assert abs(event_f1(ref, hyp, 20, 0.05) - 1.0) < 1e-9    # within collar
    assert event_f1(ref, hyp, 20, 0.01) == 0.0              # outside collar


def test_event_f1_miss_and_fp():
    ref = [np.array([1, 1, 0, 0, 0, 0])]
    hyp = [np.array([0, 0, 0, 0, 1, 1])]
    assert event_f1(ref, hyp, 20, 0.02) == 0.0


def test_clip_correct_both_right():
    rs = [np.array([0, 1, 1, 0])]; hs = [np.array([0, 1, 1, 0])]
    re = [np.array([1, 1, 0, 0])]; he = [np.array([1, 1, 0, 0])]
    assert clip_correct_rate(rs, hs, re, he, 20, 0.05) == 1.0


def test_clip_correct_env_wrong():
    rs = [np.array([0, 1, 1, 0])]; hs = [np.array([0, 1, 1, 0])]
    re = [np.array([1, 1, 0, 0])]; he = [np.array([0, 0, 1, 1])]
    assert clip_correct_rate(rs, hs, re, he, 20, 0.02) == 0.0


def test_clip_correct_partial():
    rs = [np.array([0, 1, 0]), np.array([1, 1, 0])]
    hs = [np.array([0, 1, 0]), np.array([1, 1, 0])]
    re = [np.array([1, 0, 0]), np.array([0, 0, 1])]
    he = [np.array([1, 0, 0]), np.array([1, 0, 0])]
    assert abs(clip_correct_rate(rs, hs, re, he, 20, 0.02) - 0.5) < 1e-9


def test_utterance_label_aggregation():
    assert frames_to_utterance_label(np.array([0, 1, 0]), np.zeros(3)) == "spoof_bonafide"
    assert frames_to_utterance_label(np.zeros(3), np.array([1, 0, 0])) == "bonafide_spoof"
    assert frames_to_utterance_label(np.zeros(3), np.zeros(3)) == "bonafide_bonafide"
    assert frames_to_utterance_label(np.zeros(3), np.zeros(3), is_original=True) == "original"


def test_macro_f1():
    ref = ["spoof_bonafide", "bonafide_spoof", "original", "spoof_spoof"]
    assert abs(macro_f1(ref, ref) - 1.0) < 1e-9
    hyp = ["spoof_bonafide", "bonafide_spoof", "original", "bonafide_bonafide"]
    assert 0.0 < macro_f1(ref, hyp) < 1.0
