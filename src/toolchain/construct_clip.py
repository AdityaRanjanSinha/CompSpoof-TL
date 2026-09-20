"""
src/toolchain/construct_clip.py  --  Toolchain v1 orchestrator (Week 4, S1).

Builds ONE CompSpoof-TL clip end-to-end from REAL CompSpoofV2 sources, using the
v0 building blocks (splicer / crossfade / loudness / mixer). Produces:
  - the final 4 s, 16 kHz mono mixture,
  - two per-component frame-level label masks (speech, environment) at 20 ms,
  - a provenance record (sources, regions, SNR, seed) for the manifest.

Design (follows the construction protocol in the plan + the acoustic-shortcut finding):
  * A component track = a bonafide base with spoofed regions spliced in (if that
    component is spoofed). Regions: 1-3 per manipulated component, 0.4-2.0 s each,
    non-overlapping, boundaries on the 20 ms grid. Handled by his splice_segment.
  * Loudness/energy equalization (control C3) is applied INSIDE splice_segment
    (match_loudness), which our Week-2 audit showed is essential (speech 66% / env
    72% acoustic leak in raw sources).
  * The two component tracks are mixed at a target SNR sampled from the config range.
  * Anti-shortcut control C1 (matched bonafide-into-bonafide twin) is supported via
    build_matched_pair(): same region statistics, but the spliced-in segment is a
    DIFFERENT bonafide source instead of a spoofed one -> splice presence carries no
    label info.

This is v1: runs on real audio, one clip at a time. Batch generation (50-60k clips)
comes later once the construction spec is signed off (GNG-1).
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import numpy as np

from .splicer import splice_segment
from .mixer import mix_audio_snr

SR = 16000
FRAME_MS = 20
CLIP_SEC = 4.0
TOTAL_FRAMES = int(CLIP_SEC * 1000 / FRAME_MS)   # 200


# --------------------------------------------------------------------------- #
# region sampling
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    start_sec: float
    end_sec: float


def sample_regions(rng: np.random.Generator,
                   n_min: int = 1, n_max: int = 3,
                   dur_min: float = 0.4, dur_max: float = 2.0,
                   clip_sec: float = CLIP_SEC,
                   quantize_ms: int = FRAME_MS) -> List[Region]:
    """Sample 1-3 non-overlapping regions, each 0.4-2.0 s, quantized to the 20 ms grid."""
    q = quantize_ms / 1000.0
    n = int(rng.integers(n_min, n_max + 1))
    regions: List[Region] = []
    attempts = 0
    while len(regions) < n and attempts < 200:
        attempts += 1
        dur = float(rng.uniform(dur_min, dur_max))
        dur = round(dur / q) * q                      # snap duration to grid
        if dur < dur_min:
            continue
        start = float(rng.uniform(0.0, clip_sec - dur))
        start = round(start / q) * q                  # snap start to grid
        end = round((start + dur) / q) * q
        if end > clip_sec:
            continue
        # reject overlaps
        if any(not (end <= r.start_sec or start >= r.end_sec) for r in regions):
            continue
        regions.append(Region(start, end))
    regions.sort(key=lambda r: r.start_sec)
    return regions


# --------------------------------------------------------------------------- #
# build one component track (bonafide base + spliced spoofed regions)
# --------------------------------------------------------------------------- #
def build_component_track(base_bonafide: np.ndarray,
                          replacement: Optional[np.ndarray],
                          regions: List[Region],
                          crossfade_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (track_audio, frame_mask). If `replacement` is None or regions is empty,
    the track is the untouched bonafide base and the mask is all zeros (fully genuine).
    Otherwise each region is spliced from `replacement` and the masks are OR-ed together.
    """
    total_samples = int(CLIP_SEC * SR)
    track = np.zeros(total_samples, dtype=np.float32)
    track[:min(total_samples, len(base_bonafide))] = base_bonafide[:total_samples]
    mask = np.zeros(TOTAL_FRAMES, dtype=np.int32)

    if replacement is None or not regions:
        return track, mask

    for r in regions:
        track, m = splice_segment(
            base_audio=track,
            replacement_audio=replacement,
            start_sec=r.start_sec, end_sec=r.end_sec,
            crossfade_ms=crossfade_ms, sr=SR,
            frame_ms=FRAME_MS, total_duration_sec=CLIP_SEC,
        )
        mask = np.maximum(mask, m)     # OR the region masks together
    return track, mask


# --------------------------------------------------------------------------- #
# the clip result
# --------------------------------------------------------------------------- #
@dataclass
class ClipResult:
    audio: np.ndarray                       # final mixture (not serialized to manifest)
    speech_mask: np.ndarray                 # (200,) 1 = speech spoofed at that frame
    env_mask: np.ndarray                    # (200,) 1 = env spoofed at that frame
    provenance: dict = field(default_factory=dict)


def construct_clip(
    speech_base: np.ndarray,                # bonafide speech source
    env_base: np.ndarray,                   # bonafide environment source
    snr_db: float,
    seed: int,
    speech_spoof: Optional[np.ndarray] = None,   # spoofed speech source (if speech manipulated)
    env_spoof: Optional[np.ndarray] = None,      # spoofed env source (if env manipulated)
    crossfade_ms: float = 30.0,
    region_cfg: Optional[dict] = None,
) -> ClipResult:
    """
    Build one clip. A component is 'manipulated' iff its *_spoof array is provided.
    speech_spoof=None & env_spoof=None  -> bonafide_bonafide (fully genuine mixture).
    """
    rng = np.random.default_rng(seed)
    cfg = region_cfg or {}

    sp_regions = sample_regions(rng, **cfg) if speech_spoof is not None else []
    en_regions = sample_regions(rng, **cfg) if env_spoof   is not None else []

    speech_track, speech_mask = build_component_track(speech_base, speech_spoof, sp_regions, crossfade_ms)
    env_track,    env_mask    = build_component_track(env_base,    env_spoof,    en_regions, crossfade_ms)

    mixture = mix_audio_snr(speech_track, env_track, snr_db=snr_db)

    prov = {
        "seed": int(seed),
        "snr_db": float(snr_db),
        "crossfade_ms": float(crossfade_ms),
        "speech_manipulated": speech_spoof is not None,
        "env_manipulated": env_spoof is not None,
        "speech_regions": [asdict(r) for r in sp_regions],
        "env_regions": [asdict(r) for r in en_regions],
        "speech_spoof_frac": float(speech_mask.mean()),
        "env_spoof_frac": float(env_mask.mean()),
    }
    return ClipResult(mixture.astype(np.float32), speech_mask, env_mask, prov)


# --------------------------------------------------------------------------- #
# anti-shortcut control C1: matched bonafide-into-bonafide twin
# --------------------------------------------------------------------------- #
def build_matched_bonafide(
    speech_base: np.ndarray, env_base: np.ndarray,
    speech_bonafide_alt: Optional[np.ndarray], env_bonafide_alt: Optional[np.ndarray],
    snr_db: float, seed: int, crossfade_ms: float = 30.0,
    region_cfg: Optional[dict] = None,
) -> ClipResult:
    """
    C1 twin: identical region statistics (same seed -> same sampled regions), but the
    spliced-in segment is a DIFFERENT BONAFIDE source, not a spoofed one. The audio has
    splices, but its true label is bonafide -> 'splice present' cannot predict the label.
    Masks are returned as all-zero (ground-truth authenticity is bonafide), while the
    provenance records that splices were applied (for the shortcut-probe / C1 audit).
    """
    res = construct_clip(
        speech_base, env_base, snr_db, seed,
        speech_spoof=speech_bonafide_alt, env_spoof=env_bonafide_alt,
        crossfade_ms=crossfade_ms, region_cfg=region_cfg,
    )
    # override masks to zero: this clip is BONAFIDE despite having splices (that's the point)
    res.speech_mask = np.zeros(TOTAL_FRAMES, dtype=np.int32)
    res.env_mask = np.zeros(TOTAL_FRAMES, dtype=np.int32)
    res.provenance["c1_matched_bonafide"] = True
    res.provenance["true_label"] = "bonafide_bonafide"
    return res
