"""
CompSpoof-TL Audio Construction Toolchain (Week 3 Deliverable).
Implements raised-cosine splicing, BS.1770 loudness matching,
constant-trajectory SNR mixing, and frame label generation.
"""

from .crossfade import raised_cosine_crossfade, get_crossfade_windows
from .loudness import measure_integrated_loudness, match_loudness
from .splicer import splice_segment, quantize_time_to_frames
from .mixer import mix_audio_snr

__all__ = [
    "raised_cosine_crossfade",
    "get_crossfade_windows",
    "measure_integrated_loudness",
    "match_loudness",
    "splice_segment",
    "quantize_time_to_frames",
    "mix_audio_snr",
]
