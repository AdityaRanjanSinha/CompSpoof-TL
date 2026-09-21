"""
src/metrics/evaluate.py  --  run the FULL CompSpoof-TL metric suite in one call.

Given, per clip, the ground-truth per-component frame masks and the model's per-component
frame SCORES, produces every metric in the config list:
  frame_eer / frame_f1  (per component, per resolution 20/40/160/640 ms)
  range_eer             (per component)  <- headline
  event_f1              (per component)
  clip_correct          (joint, both components)
  macro_f1              (5-class utterance-level, ESDD2 backward-compatible)

Scores are thresholded at each metric's needs; range/event/clip use a fixed operating
threshold (default 0.5) for the hard-decision metrics, EER is threshold-free.
"""
from typing import Dict, List
import numpy as np

from .frame_metrics import frame_eer_multi_res, frame_f1
from .range_eer import range_eer, mask_to_ranges, scores_to_ranges
from .segment_metrics import (event_f1, clip_correct_rate,
                              frames_to_utterance_label, macro_f1)


def evaluate_suite(sp_scores: List[np.ndarray], sp_masks: List[np.ndarray],
                   en_scores: List[np.ndarray], en_masks: List[np.ndarray],
                   frame_ms: int = 20,
                   resolutions_ms=(20, 40, 160, 640),
                   threshold: float = 0.5,
                   collar_sec: float = 0.2) -> Dict:
    out: Dict = {"speech": {}, "environment": {}, "joint": {}, "utterance": {}}

    for comp, scores, masks in [("speech", sp_scores, sp_masks),
                                ("environment", en_scores, en_masks)]:
        c = out[comp]
        c["frame_eer"] = frame_eer_multi_res(scores, masks, frame_ms, resolutions_ms)
        c["frame_f1"] = float(np.mean([
            frame_f1(s, m, threshold) for s, m in zip(scores, masks)]))
        c["range_eer"] = range_eer(
            [mask_to_ranges(m, frame_ms) for m in masks],
            [scores_to_ranges(s, frame_ms) for s in scores])[0]
        hyp_masks = [(s >= threshold).astype(int) for s in scores]
        c["event_f1"] = event_f1(masks, hyp_masks, frame_ms, collar_sec)

    sp_hyp = [(s >= threshold).astype(int) for s in sp_scores]
    en_hyp = [(s >= threshold).astype(int) for s in en_scores]
    out["joint"]["clip_correct"] = clip_correct_rate(
        sp_masks, sp_hyp, en_masks, en_hyp, frame_ms, collar_sec)

    ref_lab = [frames_to_utterance_label(sm, em) for sm, em in zip(sp_masks, en_masks)]
    hyp_lab = [frames_to_utterance_label(sh, eh) for sh, eh in zip(sp_hyp, en_hyp)]
    out["utterance"]["macro_f1"] = macro_f1(ref_lab, hyp_lab)
    return out
