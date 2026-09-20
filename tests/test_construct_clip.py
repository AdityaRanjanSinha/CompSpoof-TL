"""Unit tests for the toolchain v1 orchestrator (construct_clip)."""
import numpy as np
from src.toolchain.construct_clip import (
    construct_clip, build_matched_bonafide, sample_regions, TOTAL_FRAMES, SR, CLIP_SEC,
)

N = int(CLIP_SEC * SR)
_rng = np.random.default_rng(0)
def _tone(f, amp=0.3):
    t = np.arange(N) / SR
    return (amp*np.sin(2*np.pi*f*t) + 0.02*_rng.standard_normal(N)).astype(np.float32)

SP_B, SP_S = _tone(180), _tone(190)      # bonafide / spoofed speech
EN_B, EN_S = _tone(500), _tone(520)      # bonafide / spoofed environment
CFG = dict(n_min=1, n_max=3, dur_min=0.4, dur_max=2.0)


def test_region_sampling_constraints():
    for s in range(20):
        regs = sample_regions(np.random.default_rng(s), **CFG)
        assert 1 <= len(regs) <= 3
        for r in regs:
            assert 0.4 - 1e-6 <= (r.end_sec - r.start_sec) <= 2.0 + 1e-6
            assert 0 <= r.start_sec < r.end_sec <= CLIP_SEC
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                a, b = regs[i], regs[j]
                assert a.end_sec <= b.start_sec or b.end_sec <= a.start_sec


def test_bonafide_bonafide_masks_zero():
    r = construct_clip(SP_B, EN_B, snr_db=9.0, seed=1)
    assert r.audio.shape == (N,)
    assert r.speech_mask.shape == (TOTAL_FRAMES,)
    assert r.speech_mask.sum() == 0 and r.env_mask.sum() == 0


def test_speech_spoofed_only():
    r = construct_clip(SP_B, EN_B, snr_db=8.0, seed=2, speech_spoof=SP_S)
    assert r.speech_mask.sum() > 0
    assert r.env_mask.sum() == 0


def test_both_spoofed():
    r = construct_clip(SP_B, EN_B, snr_db=10.0, seed=3, speech_spoof=SP_S, env_spoof=EN_S)
    assert r.speech_mask.sum() > 0 and r.env_mask.sum() > 0


def test_mask_aligns_with_regions():
    r = construct_clip(SP_B, EN_B, snr_db=9.0, seed=7, speech_spoof=SP_S)
    for reg in r.provenance["speech_regions"]:
        sf = int(round(reg["start_sec"] * 1000 / 20))
        ef = int(round(reg["end_sec"] * 1000 / 20))
        assert r.speech_mask[sf:ef].all()


def test_determinism():
    a = construct_clip(SP_B, EN_B, 9.0, 42, speech_spoof=SP_S)
    b = construct_clip(SP_B, EN_B, 9.0, 42, speech_spoof=SP_S)
    assert np.array_equal(a.speech_mask, b.speech_mask)
    assert np.allclose(a.audio, b.audio)


def test_c1_matched_bonafide_twin():
    r = build_matched_bonafide(SP_B, EN_B, speech_bonafide_alt=_tone(182),
                               env_bonafide_alt=None, snr_db=9.0, seed=2)
    assert r.speech_mask.sum() == 0 and r.env_mask.sum() == 0   # label is bonafide
    assert r.provenance["c1_matched_bonafide"]
    assert len(r.provenance["speech_regions"]) > 0              # but splices ARE present
    # same seed -> identical region statistics to the spoofed version (matched control)
    r3 = construct_clip(SP_B, EN_B, 9.0, 2, speech_spoof=SP_S)
    a = [(x["start_sec"], x["end_sec"]) for x in r.provenance["speech_regions"]]
    b = [(x["start_sec"], x["end_sec"]) for x in r3.provenance["speech_regions"]]
    assert a == b
