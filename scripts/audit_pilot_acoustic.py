"""
scripts/audit_pilot_acoustic.py  --  C4 verification: acoustic shortcut audit on the
GENERATED CompSpoof-TL pilot (not the source data).

This is the payoff of the Week-2 finding. The SOURCE CompSpoofV2 clips leaked the
component classes acoustically (speech ~66%, env ~72%). This script runs the SAME
acoustic-only audit on the clips WE constructed, to check whether the loudness/energy
equalization (control C3) applied during splicing removed the leak.

  PASS (construction is sound): acoustic-only accuracy near chance
        (4 mixed classes ~0.25 ; speech/env spoof-vs-bonafide ~0.50).
  FAIL: leak persists -> C3 insufficient (likely spectral, not just level) -> revise
        construction or report as a known limitation.

Reads the pilot manifest (clip_id, label) and audio from <pilot>/audio/<clip_id>.wav.
Excludes the C1 twins from the class-separation test (they are bonafide by construction).

Usage:
    python scripts/audit_pilot_acoustic.py --pilot ~/scratch/CompSpoof-TL/pilot
"""
import argparse, os
import numpy as np
import pandas as pd
import soundfile as sf
import pyloudnorm as pyln
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

SEED = 13
MIXED = ["bonafide_bonafide", "bonafide_spoof", "spoof_bonafide", "spoof_spoof"]
ACOUSTIC = ["duration_sec", "sample_peak", "rms", "loudness_lufs",
            "mean_val", "std_val", "zcr"]


def feats(path):
    try:
        y, sr = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y**2)))
        m = pyln.Meter(sr)
        try:
            loud = float(m.integrated_loudness(y.astype(np.float64)))
            if not np.isfinite(loud): loud = -70.0
        except Exception:
            loud = -70.0
        return dict(duration_sec=len(y)/sr, sample_peak=peak, rms=rms,
                    loudness_lufs=loud, mean_val=float(np.mean(y)),
                    std_val=float(np.std(y)), zcr=float(np.mean((y[:-1]*y[1:])<0)))
    except Exception:
        return None


def evaluate(X, y, chance, title, out):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    Xs = StandardScaler().fit_transform(X)
    rf = cross_val_score(RandomForestClassifier(n_estimators=200, max_depth=8,
                         random_state=SEED, n_jobs=-1), X, y, cv=skf, n_jobs=-1).mean()
    lr = cross_val_score(LogisticRegression(max_iter=1000), Xs, y, cv=skf).mean()
    best = max(rf, lr); margin = best - chance
    verdict = ("PASS — near chance; construction removed the acoustic leak."
               if margin < 0.10 else
               "FAIL — leak persists; C3 (loudness) insufficient, likely spectral.")
    out += ["-"*60, title,
            f"  n={len(y)}  chance={chance:.3f}  RF={rf:.3f}  LR={lr:.3f}",
            f"  --> best {best:.3f}  (margin {margin:+.3f})   {verdict}"]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", default="pilot_acoustic_audit_report.txt")
    a = ap.parse_args()
    pilot = a.pilot
    man = pd.read_csv(os.path.join(pilot, "manifest.csv"))
    # exclude C1 twins from the leak test (they're bonafide by construction)
    man = man[~man["clip_id"].str.contains("c1twin")].copy()
    man = man[man["label"].isin(MIXED)].copy()

    rows = []
    for _, r in man.iterrows():
        p = os.path.join(pilot, "audio", r["clip_id"] + ".wav")
        f = feats(p)
        if f:
            f["label"] = r["label"]
            rows.append(f)
    df = pd.DataFrame(rows)
    df[ACOUSTIC] = df[ACOUSTIC].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df[ACOUSTIC]
    print(f"Audited {len(df)} generated clips.\n")

    out = ["="*60,
           "ACOUSTIC SHORTCUT AUDIT ON GENERATED PILOT (C4 verification)",
           "Compare to SOURCE-data leak: speech ~0.66, env ~0.72, 4-class ~0.49",
           "="*60]
    evaluate(X, df["label"], 0.25, "4 mixed classes (chance 0.25)", out)
    evaluate(X, df["label"].str.split("_").str[0], 0.50,
             "SPEECH spoof-vs-bonafide (chance 0.50)", out)
    evaluate(X, df["label"].str.split("_").str[1], 0.50,
             "ENV spoof-vs-bonafide (chance 0.50)", out)
    out += ["="*60,
            "If these are near chance AND well below the source-data numbers,",
            "the construction's loudness/energy equalization (C3) is working.",
            "="*60]
    rep = "\n".join(out)
    print("\n"+rep)
    open(a.out, "w").write(rep+"\n")
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
