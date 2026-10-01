"""
src/data/dataset.py  --  CompSpoof-TL benchmark loader.

Layout written by scripts/generate_benchmark_v2.py:
    <root>/<split>/manifest.csv
    <root>/<split>/audio/<clip_id>.wav            4 s, 16 kHz mono
    <root>/<split>/masks/<clip_id>_speech.npy      (200,) 0/1, 1 = spoofed 20 ms frame
    <root>/<split>/masks/<clip_id>_env.npy         (200,)

Each item: wav float32 (64000,), speech_mask float32 (200,), env_mask float32 (200,), clip_id.
"""
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


def _fix_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


class CompSpoofTLDataset(Dataset):
    def __init__(self, root: str, split: str, limit: Optional[int] = None, seed: int = 0):
        self.dir = Path(root) / split
        df = pd.read_csv(self.dir / "manifest.csv", dtype={"clip_id": str}, low_memory=False)
        if limit is not None and limit < len(df):
            df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
        self.df = df
        self.ids = df["clip_id"].tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        wav, sr = sf.read(self.dir / "audio" / f"{cid}.wav", dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == SR, f"{cid}: sample rate {sr} != {SR}"
        wav = _fix_length(wav, CLIP_SAMPLES)
        sm = _fix_length(np.load(self.dir / "masks" / f"{cid}_speech.npy").astype(np.float32), N_FRAMES)
        em = _fix_length(np.load(self.dir / "masks" / f"{cid}_env.npy").astype(np.float32), N_FRAMES)
        return torch.from_numpy(wav), torch.from_numpy(sm), torch.from_numpy(em), cid

    def pos_weights(self, max_items: int = 5000, seed: int = 0):
        """neg/pos frame ratio per component (for BCE pos_weight), from a sample of masks."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.ids), size=min(max_items, len(self.ids)), replace=False)
        sp = en = 0.0
        for i in idx:
            cid = self.ids[i]
            sp += np.load(self.dir / "masks" / f"{cid}_speech.npy").mean()
            en += np.load(self.dir / "masks" / f"{cid}_env.npy").mean()
        sp, en = sp / len(idx), en / len(idx)
        return (1 - sp) / max(sp, 1e-6), (1 - en) / max(en, 1e-6)
