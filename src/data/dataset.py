"""
src/data/dataset.py  --  CompSpoof-TL benchmark loader.

Layout written by scripts/generate_benchmark_v2.py:
    <root>/<split>/manifest.csv
    <root>/<split>/audio/<clip_id>.wav            4 s, 16 kHz mono
    <root>/<split>/masks/<clip_id>_speech.npy      (200,) 0/1, 1 = spoofed 20 ms frame
    <root>/<split>/masks/<clip_id>_env.npy         (200,)

Each item: wav float32 (64000,), speech_mask float32 (200,), env_mask float32 (200,), clip_id.

C4 shortcut probe support:
    subset="c1"          keep only the C1 matched-bonafide clips (all-bonafide splices)
    label_mode="regions" labels = the SPLICE regions from the manifest (speech_regions /
                         env_regions) instead of the spoof masks (which are all-zero for C1).
"""
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000
CLIP_SAMPLES = 4 * SR
N_FRAMES = 200
FRAME_SEC = 0.02
_NUM = re.compile(r"\d*\.?\d+")   # unsigned: "0.4-1.2" must not read as -1.2


def _fix_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def regions_to_mask(cell, n_frames: int = N_FRAMES) -> np.ndarray:
    """
    Parse a manifest regions cell into a (n_frames,) 0/1 mask. Accepts any text holding
    start/end pairs, e.g. "[(0.4, 1.2), (2.0, 2.6)]", "0.40-1.20;2.00-2.60", "[[0.4,1.2]]".
    Units are inferred: all values <= 4.0 -> seconds; integers <= n_frames -> frames;
    otherwise milliseconds.
    """
    m = np.zeros(n_frames, np.float32)
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return m
    vals = [float(x) for x in _NUM.findall(str(cell))]
    if len(vals) < 2:
        return m
    vals = vals[: len(vals) // 2 * 2]
    if max(vals) <= 4.0 + 1e-6:
        to_frame = lambda v: v / FRAME_SEC
    elif all(float(v).is_integer() for v in vals) and max(vals) <= n_frames:
        to_frame = lambda v: v
    else:
        to_frame = lambda v: v / 1000.0 / FRAME_SEC
    for s, e in zip(vals[0::2], vals[1::2]):
        a, b = int(round(to_frame(s))), int(round(to_frame(e)))
        m[max(0, a):min(n_frames, b)] = 1.0
    return m


class CompSpoofTLDataset(Dataset):
    def __init__(self, root: str, split: str, limit: Optional[int] = None, seed: int = 0,
                 subset: str = "all", label_mode: str = "mask"):
        self.dir = Path(root) / split
        df = pd.read_csv(self.dir / "manifest.csv", dtype={"clip_id": str}, low_memory=False)
        if subset == "c1":
            df = df[df["c1_matched_bonafide"].map(_truthy)].reset_index(drop=True)
            if len(df) == 0:
                raise RuntimeError(f"{split}: no rows with c1_matched_bonafide=True")
        if limit is not None and limit < len(df):
            df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
        self.df = df
        self.ids = df["clip_id"].tolist()
        self.label_mode = label_mode
        if label_mode == "regions":
            self.reg = {c: [regions_to_mask(v) for v in df[f"{c}_regions"]] for c in ("speech", "env")}
            frac = {c: float(np.mean([r.mean() for r in self.reg[c]])) for c in ("speech", "env")}
            print(f"[{split}/{subset}] region labels: {len(df)} clips, mean labelled-frame "
                  f"fraction speech {frac['speech']:.3f} env {frac['env']:.3f}", flush=True)
            if frac["speech"] == 0 and frac["env"] == 0:
                raise RuntimeError("could not parse any splice regions -- check the *_regions columns")

    def __len__(self):
        return len(self.ids)

    def masks(self, i):
        if self.label_mode == "regions":
            return self.reg["speech"][i], self.reg["env"][i]
        cid = self.ids[i]
        sm = _fix_length(np.load(self.dir / "masks" / f"{cid}_speech.npy").astype(np.float32), N_FRAMES)
        em = _fix_length(np.load(self.dir / "masks" / f"{cid}_env.npy").astype(np.float32), N_FRAMES)
        return sm, em

    def __getitem__(self, i):
        cid = self.ids[i]
        wav, sr = sf.read(self.dir / "audio" / f"{cid}.wav", dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == SR, f"{cid}: sample rate {sr} != {SR}"
        wav = _fix_length(wav, CLIP_SAMPLES)
        sm, em = self.masks(i)
        return torch.from_numpy(wav), torch.from_numpy(sm), torch.from_numpy(em), cid

    def pos_weights(self, max_items: int = 5000, seed: int = 0):
        """neg/pos frame ratio per component (for BCE pos_weight), from a sample of labels."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.ids), size=min(max_items, len(self.ids)), replace=False)
        sp = en = 0.0
        for i in idx:
            sm, em = self.masks(i)
            sp += sm.mean(); en += em.mean()
        sp, en = sp / len(idx), en / len(idx)
        return (1 - sp) / max(sp, 1e-6), (1 - en) / max(en, 1e-6)
