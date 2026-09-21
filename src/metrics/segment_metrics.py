"""
src/metrics/segment_metrics.py  --  Event-based F1, joint clip-correct rate, and
utterance-level Macro-F1 for CompSpoof-TL.

Completes the metric suite alongside frame_metrics.py and range_eer.py.
Config metric list: [macro_f1, frame_eer, frame_f1, range_eer, event_f1, clip_correct].

Conventions (consistent with the rest of the suite):
  - a per-component prediction is a 0/1 frame mask (thresholded scores)
  - ranges are (start_sec, end_sec) spoof *events*
  - higher score = more likely spoof
"""
from typing import List, Tuple, Dict
import numpy as np

Event = Tuple[float, float]     # (start_sec, end_sec) of a spoof event


# --------------------------------------------------------------------------- #
# mask -> list of spoof events (contiguous 1-runs)
# --------------------------------------------------------------------------- #
def mask_to_events(mask: np.ndarray, frame_ms: int = 20) -> List[Event]:
    """Contiguous runs of 1 become spoof events (start_sec, end_sec)."""
    mask = np.asarray(mask).astype(int)
    events, i, n = [], 0, len(mask)
    while i < n:
        if mask[i] == 1:
            j = i
            while j < n and mask[j] == 1:
                j += 1
            events.append((i * frame_ms / 1000.0, j * frame_ms / 1000.0))
            i = j
        else:
            i += 1
    return events


# --------------------------------------------------------------------------- #
# event-based F1 with an onset collar
# --------------------------------------------------------------------------- #
def _onset_match(ref_ev: Event, hyp_ev: Event, collar: float) -> bool:
    """A predicted event matches a reference event if their ONSETS are within `collar`."""
    return abs(ref_ev[0] - hyp_ev[0]) <= collar


def event_f1(ref_masks: List[np.ndarray],
             hyp_masks: List[np.ndarray],
             frame_ms: int = 20,
             collar_sec: float = 0.2) -> float:
    """
    Event-based F1 with an onset collar (sound-event-detection style, PSDS-like).
    A reference event is a TRUE POSITIVE if some predicted event's onset lies within
    `collar` of it (1-to-1, greedy). Unmatched refs -> FN, unmatched hyps -> FP.
    Aggregated over all clips (micro-average).
    """
    tp = fp = fn = 0
    for rm, hm in zip(ref_masks, hyp_masks):
        ref = mask_to_events(rm, frame_ms)
        hyp = mask_to_events(hm, frame_ms)
        used_hyp = set()
        for r in ref:
            hit = None
            for k, h in enumerate(hyp):
                if k in used_hyp:
                    continue
                if _onset_match(r, h, collar_sec):
                    hit = k
                    break
            if hit is not None:
                tp += 1
                used_hyp.add(hit)
            else:
                fn += 1
        fp += len(hyp) - len(used_hyp)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


# --------------------------------------------------------------------------- #
# joint clip-correct rate
# --------------------------------------------------------------------------- #
def clip_correct_rate(ref_speech: List[np.ndarray], hyp_speech: List[np.ndarray],
                      ref_env: List[np.ndarray], hyp_env: List[np.ndarray],
                      frame_ms: int = 20, collar_sec: float = 0.2) -> float:
    """
    Fraction of clips where BOTH components' predicted spoof-event sets match ground truth
    (same #events, each matched under the onset collar). This is the strict 'deployment'
    number: a clip is correct only if speech AND environment are both localized correctly.
    """
    def component_correct(ref_mask, hyp_mask):
        ref = mask_to_events(ref_mask, frame_ms)
        hyp = mask_to_events(hyp_mask, frame_ms)
        if len(ref) != len(hyp):
            return False
        used = set()
        for r in ref:
            hit = None
            for k, h in enumerate(hyp):
                if k in used:
                    continue
                if _onset_match(r, h, collar_sec):
                    hit = k; break
            if hit is None:
                return False
            used.add(hit)
        return True

    n = len(ref_speech)
    if n == 0:
        return 0.0
    correct = 0
    for rs, hs, rev, hev in zip(ref_speech, hyp_speech, ref_env, hyp_env):
        if component_correct(rs, hs) and component_correct(rev, hev):
            correct += 1
    return correct / n


# --------------------------------------------------------------------------- #
# utterance-level Macro-F1 (5-class, ESDD2 backward-compatible)
# --------------------------------------------------------------------------- #
def frames_to_utterance_label(speech_mask: np.ndarray, env_mask: np.ndarray,
                              is_original: bool = False) -> str:
    """
    Aggregate per-component frame masks into the 5-class utterance label (so our benchmark
    stays backward-compatible with ESDD2 scoring). A component is 'spoof' at the utterance
    level if ANY of its frames are spoofed.
    """
    if is_original:
        return "original"
    sp = "spoof" if np.asarray(speech_mask).sum() > 0 else "bonafide"
    en = "spoof" if np.asarray(env_mask).sum() > 0 else "bonafide"
    return f"{sp}_{en}"


def macro_f1(ref_labels: List[str], hyp_labels: List[str]) -> float:
    """5-class Macro-F1 (unweighted mean of per-class F1), as ESDD2 reports."""
    from sklearn.metrics import f1_score
    classes = sorted(set(ref_labels) | set(hyp_labels))
    return float(f1_score(ref_labels, hyp_labels, labels=classes,
                          average="macro", zero_division=0))
