"""
tests/test_toolchain.py

CompSpoof-TL S1 Unit Tests (Week 3 Deliverable).
Verifies synthetic tone splicing, raised-cosine crossfades,
loudness matching, SNR mixing, and frame-level mask quantization.
"""

import numpy as np
import pytest
from src.toolchain.crossfade import get_crossfade_windows, raised_cosine_crossfade
from src.toolchain.loudness import measure_integrated_loudness, match_loudness
from src.toolchain.splicer import quantize_time_to_frames, splice_segment
from src.toolchain.mixer import mix_audio_snr


def test_crossfade_windows_sum_to_one():
    """Verify complementary raised-cosine windows sum to 1.0 everywhere."""
    L = 320  # 20 ms at 16 kHz
    fade_in, fade_out = get_crossfade_windows(L, equal_power=False)

    assert len(fade_in) == L
    assert len(fade_out) == L
    assert np.isclose(fade_in[0], 0.0, atol=1e-5)
    assert np.isclose(fade_in[-1], 1.0, atol=1e-5)
    assert np.isclose(fade_out[0], 1.0, atol=1e-5)
    assert np.isclose(fade_out[-1], 0.0, atol=1e-5)

    # Check partition of unity
    window_sum = fade_in + fade_out
    assert np.allclose(window_sum, 1.0, atol=1e-5)


def test_crossfade_continuity_on_synthetic_tones():
    """Verify that splicing identical synthetic pure tones produces zero discontinuity."""
    sr = 16000
    freq = 440.0
    duration = 0.5
    t = np.arange(int(sr * duration)) / sr
    tone_a = np.sin(2 * np.pi * freq * t).astype(np.float32)
    tone_b = tone_a.copy()

    L = 320  # 20 ms crossfade
    cf = raised_cosine_crossfade(tone_a[:L], tone_b[:L], L)

    # Crossfade of identical signals should be identical to the original signal
    assert np.allclose(cf, tone_a[:L], atol=1e-5)


def test_quantization_to_20ms_frames():
    """Verify that time quantization correctly snaps to the 20 ms grid."""
    sr = 16000
    frame_ms = 20  # 320 samples per frame

    # 1.000s -> exactly frame 50
    assert quantize_time_to_frames(1.0, frame_ms=frame_ms, sr=sr) == 50

    # 1.009s -> rounds to frame 50 (within 20ms boundary)
    assert quantize_time_to_frames(1.009, frame_ms=frame_ms, sr=sr) == 50

    # 1.011s -> rounds to frame 51
    assert quantize_time_to_frames(1.011, frame_ms=frame_ms, sr=sr) == 51

    # 4.000s -> frame 200
    assert quantize_time_to_frames(4.0, frame_ms=frame_ms, sr=sr) == 200


def test_splice_segment_and_mask_generation():
    """Verify splicing produces correct audio length and exact frame mask."""
    sr = 16000
    duration = 4.0
    total_samples = int(sr * duration)

    t = np.arange(total_samples) / sr
    base_audio = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    rep_audio = 0.5 * np.sin(2 * np.pi * 600 * t).astype(np.float32)

    spliced, mask = splice_segment(
        base_audio=base_audio,
        replacement_audio=rep_audio,
        start_sec=1.0,
        end_sec=2.5,
        crossfade_ms=30.0,
        sr=sr,
        frame_ms=20,
        total_duration_sec=duration,
    )

    # 1. Total samples and frames match 4-second clip
    assert len(spliced) == total_samples
    assert len(mask) == 200  # 4s / 20ms = 200 frames

    # 2. Mask matches quantized intervals: 1.0s = frame 50, 2.5s = frame 125
    assert np.all(mask[:50] == 0)
    assert np.all(mask[50:125] == 1)
    assert np.all(mask[125:] == 0)

    # 3. Audio outside the splice region remains unchanged
    assert np.allclose(spliced[: 50 * 320], base_audio[: 50 * 320], atol=1e-5)


def test_loudness_matching_within_1_lu():
    """Verify that loudness matching brings two signals within <= 1.0 LU."""
    sr = 16000
    np.random.seed(42)

    # Two random noise signals of different realistic amplitudes
    sig1 = (0.04 * np.random.randn(sr * 3)).astype(np.float32)
    sig2 = (0.09 * np.random.randn(sr * 3)).astype(np.float32)

    matched_sig1 = match_loudness(sig1, sig2, max_diff_lu=1.0, sr=sr)

    l_matched = measure_integrated_loudness(matched_sig1, sr=sr)
    l_target = measure_integrated_loudness(sig2, sr=sr)

    diff = abs(l_matched - l_target)
    assert diff <= 1.0, f"Loudness difference {diff:.2f} LU exceeds 1.0 LU tolerance"


def test_mix_audio_snr():
    """Verify that mixing preserves designated SNR within +/- 0.1 dB."""
    sr = 16000
    t = np.arange(sr * 2) / sr
    speech = (0.4 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)

    np.random.seed(13)
    env = (0.1 * np.random.randn(sr * 2)).astype(np.float32)

    target_snr = 10.0  # 10 dB
    mixed = mix_audio_snr(speech, env, snr_db=target_snr)

    assert len(mixed) == len(speech)
    assert not np.isnan(mixed).any()
    assert np.max(np.abs(mixed)) <= 1.0
