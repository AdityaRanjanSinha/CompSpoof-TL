"""
scripts/make_listening_sheet.py  --  5% listening check of the eval split (plan, week 9):
"manually listened to, split between the two students, disagreements adjudicated by the guide,
as a sanity check that manipulations are perceptually plausible and boundaries are inaudible".

Draws a stratified 5% sample (by class label and seen/unseen), splits it in two halves, copies
the audio into one folder per listener, and writes a sheet per listener with the manipulated
regions (in seconds) and blank columns to fill in. A 10% overlap between the halves lets us
measure agreement.

Usage: python scripts/make_listening_sheet.py --bench /flash/ee/btech/ee3231024/CompSpoof-TL \
          --out /flash/ee/btech/ee1230446/listening_check
Then:  tar czf listening_check.tgz -C /flash/ee/btech/ee1230446 listening_check   (download to laptop)
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd


def secs(cell):
    try:
        r = json.loads(cell) if isinstance(cell, str) and cell.strip().startswith("[") else []
        return "; ".join(f"{d['start_sec']:.2f}-{d['end_sec']:.2f}" for d in r)
    except Exception:
        return str(cell)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--bench", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--frac", type=float, default=0.05)
    a.add_argument("--overlap", type=float, default=0.10)
    a.add_argument("--seed", type=int, default=2027)
    a.add_argument("--names", default="Aditya,Anshul")
    args = a.parse_args()
    d = Path(args.bench) / "eval"
    df = pd.read_csv(d / "manifest.csv", dtype={"clip_id": str}, low_memory=False)
    df["stratum"] = df["label"].astype(str) + "|" + df["unseen_generator"].astype(str)
    rng = np.random.default_rng(args.seed)
    pick = []
    for _, idx in df.groupby("stratum").indices.items():
        k = max(1, int(round(len(idx) * args.frac)))
        pick += list(rng.choice(idx, size=k, replace=False))
    samp = df.loc[pick].sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_ov = int(round(len(samp) * args.overlap))
    shared, rest = samp.iloc[:n_ov], samp.iloc[n_ov:]
    halves = [pd.concat([shared, rest.iloc[0::2]]), pd.concat([shared, rest.iloc[1::2]])]
    cols = ["clip_id", "label", "speech_generator", "env_generator", "unseen_generator",
            "speech_regions", "env_regions", "c1_matched_bonafide"]
    for name, h in zip(args.names.split(","), halves):
        o = Path(args.out) / name; (o / "audio").mkdir(parents=True, exist_ok=True)
        sheet = h[[c for c in cols if c in h.columns]].copy()
        sheet["speech_regions"] = sheet["speech_regions"].map(secs)
        sheet["env_regions"] = sheet["env_regions"].map(secs)
        sheet["in_overlap"] = sheet["clip_id"].isin(shared["clip_id"])
        for c in ("plausible_Y_N", "audible_click_at_boundary_Y_N", "splice_noticeable_Y_N", "notes"):
            sheet[c] = ""
        sheet.to_csv(o / f"listening_sheet_{name}.csv", index=False)
        for cid in sheet["clip_id"]:
            shutil.copy(d / "audio" / f"{cid}.wav", o / "audio" / f"{cid}.wav")
        print(f"{name}: {len(sheet)} clips ({n_ov} shared) -> {o}")
    print(f"sample: {len(samp)} clips = {len(samp)/len(df):.1%} of eval, strata: {samp['stratum'].nunique()}")


if __name__ == "__main__":
    main()
