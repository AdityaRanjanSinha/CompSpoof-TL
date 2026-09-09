"""
Raised-cosine crossfade generator.
Ensures smooth, click-free audio transitions at splice boundaries.
"""

import numpy as np


def get_crossfade_windows(length_samples: int, equal_power: bool = False):
    """
    Generate complementary fade-in and fade-out raised-cosine windows of length L.
    
    For equal amplitude (default):
        fade_in(t) = 0.5 * (1 - cos(pi * t / L))
        fade_out(t) = 0.5 * (1 + cos(pi * t / L))
        fade_in(t) + fade_out(t) == 1.0 everywhere.
    """
    if length_samples <= 0:
        raise ValueError("length_samples must be > 0")
    if length_samples == 1:
        return np.array([1.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    t = np.linspace(0, np.pi, length_samples, endpoint=True, dtype=np.float32)
    fade_in = 0.5 * (1.0 - np.cos(t))
    fade_out = 0.5 * (1.0 + np.cos(t))

    if equal_power:
        fade_in = np.sqrt(fade_in)
        fade_out = np.sqrt(fade_out)

    return fade_in.astype(np.float32), fade_out.astype(np.float32)


def raised_cosine_crossfade(signal_a: np.ndarray, signal_b: np.ndarray, length_samples: int, equal_power: bool = False) -> np.ndarray:
    """
    Crossfades from signal_a (fading out) to signal_b (fading in) over length_samples.
    """
    if len(signal_a) != length_samples or len(signal_b) != length_samples:
        raise ValueError(f"Both signals must have length {length_samples}, got {len(signal_a)} and {len(signal_b)}")

    fade_in, fade_out = get_crossfade_windows(length_samples, equal_power=equal_power)
    return (signal_a * fade_out + signal_b * fade_in).astype(np.float32)
