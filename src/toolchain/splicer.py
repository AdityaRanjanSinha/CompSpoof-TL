"""
Segment Splicer with Raised-Cosine Boundaries and Frame Quantization.
Splices a replacement (spoofed or bonafide) segment into a base audio clip,
ensuring region boundaries snap to the 20 ms frame grid (320 samples at 16 kHz).
"""

import numpy as np
from .crossfade import get_crossfade_windows
from .loudness import match_loudness


def quantize_time_to_frames(time_sec: float, frame_ms: int = 20, sr: int = 16000) -> int:
    """Quantize a time point in seconds to the nearest discrete frame index."""
    frame_samples = int(sr * (frame_ms / 1000.0))
    total_samples = int(round(time_sec * sr))
    frame_idx = int(round(total_samples / frame_samples))
    return frame_idx


def splice_segment(
    base_audio: np.ndarray,
    replacement_audio: np.ndarray,
    start_sec: float,
    end_sec: float,
    crossfade_ms: float = 30.0,
    sr: int = 16000,
    frame_ms: int = 20,
    total_duration_sec: float = 4.0,
):
    """
    Splices a segment of replacement_audio into base_audio from start_sec to end_sec.
    
    Returns:
        output_audio: np.ndarray of shape (total_samples,)
        frame_mask: np.ndarray of shape (num_frames,) where 1 marks spliced frames.
    """
    frame_samples = int(sr * (frame_ms / 1000.0))  # 320 samples for 20ms @ 16kHz
    total_samples = int(round(total_duration_sec * sr))
    total_frames = total_samples // frame_samples

    # 1. Quantize start and end times to the frame grid
    start_frame = max(0, quantize_time_to_frames(start_sec, frame_ms=frame_ms, sr=sr))
    end_frame = min(total_frames, quantize_time_to_frames(end_sec, frame_ms=frame_ms, sr=sr))

    if end_frame <= start_frame:
        raise ValueError(f"end_sec ({end_sec}s -> frame {end_frame}) must be strictly greater than start_sec ({start_sec}s -> frame {start_frame})")

    start_sample = start_frame * frame_samples
    end_sample = end_frame * frame_samples
    region_len = end_sample - start_sample

    # 2. Ensure base_audio is exactly total_samples
    base = np.zeros(total_samples, dtype=np.float32)
    copy_len = min(total_samples, len(base_audio))
    base[:copy_len] = base_audio[:copy_len]

    # 3. Ensure replacement_audio has sufficient length for the splice region
    rep = np.zeros(region_len, dtype=np.float32)
    rep_copy = min(region_len, len(replacement_audio))
    rep[:rep_copy] = replacement_audio[:rep_copy]

    # 4. Loudness match replacement segment to the base audio segment
    base_sub = base[start_sample:end_sample]
    rep_matched = match_loudness(rep, base_sub, max_diff_lu=1.0, sr=sr)

    # 5. Compute crossfade sample count
    cf_samples = int(round(sr * (crossfade_ms / 1000.0)))
    cf_samples = min(cf_samples, region_len // 2)

    output = base.copy()

    if cf_samples > 1:
        fade_in, fade_out = get_crossfade_windows(cf_samples)

        # Apply replacement in the center
        output[start_sample + cf_samples : end_sample - cf_samples] = rep_matched[cf_samples : region_len - cf_samples]

        # Crossfade at entry boundary
        b_in = base[start_sample : start_sample + cf_samples]
        r_in = rep_matched[:cf_samples]
        output[start_sample : start_sample + cf_samples] = b_in * fade_out + r_in * fade_in

        # Crossfade at exit boundary
        r_out = rep_matched[region_len - cf_samples : region_len]
        b_out = base[end_sample - cf_samples : end_sample]
        output[end_sample - cf_samples : end_sample] = r_out * fade_out + b_out * fade_in
    else:
        output[start_sample:end_sample] = rep_matched

    # 6. Generate exact binary frame mask (1 for spliced frames, 0 otherwise)
    frame_mask = np.zeros(total_frames, dtype=np.int32)
    frame_mask[start_frame:end_frame] = 1

    return output.astype(np.float32), frame_mask
