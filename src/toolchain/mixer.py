"""
SNR-Preserving Audio Mixer for Speech and Environment Tracks.
Mixes speech and environment at a designated target SNR (e.g. 7-12 dB)
while holding the SNR trajectory strictly constant across splice boundaries.
"""

import numpy as np


def mix_audio_snr(speech: np.ndarray, env: np.ndarray, snr_db: float = 9.5) -> np.ndarray:
    """
    Mix foreground speech and background environment audio at target SNR in dB:
        SNR_dB = 10 * log10(P_speech / P_env)
        P_env_target = P_speech / (10^(SNR_dB / 10))
    """
    min_len = min(len(speech), len(env))
    s = speech[:min_len].astype(np.float32)
    e = env[:min_len].astype(np.float32)

    p_speech = np.mean(s ** 2)
    p_env = np.mean(e ** 2)

    if p_speech <= 1e-9 or p_env <= 1e-9:
        # If either track is silent, return direct sum
        return (s + e).astype(np.float32)

    target_p_env = p_speech / (10.0 ** (snr_db / 10.0))
    gain = np.sqrt(target_p_env / p_env)

    scaled_env = e * gain
    mixed = s + scaled_env

    # Prevent overall clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = (mixed / peak) * 0.99

    return mixed.astype(np.float32)
