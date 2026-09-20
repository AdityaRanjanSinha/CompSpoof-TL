"""
scripts/audit_acoustic_component_shortcut.py

CompSpoof-TL S1 — Week-2 follow-up audit (completes the safety question).

WHY THIS EXISTS
---------------
The first shortcut audit (audit_metadata_shortcuts.py) reported ~55% accuracy from
ACOUSTIC-ONLY features on the full 5-class task, and concluded the leak is "primarily
original-vs-mixed" (benign). That conclusion was ASSUMED, not tested.

A back-of-envelope check shows it's suspicious: if the classifier only separated
`original` from the 4 mixed classes and was at chance among the mixed ones, expected
accuracy would be ~= P(original)*1 + P(mixed)*0.25  ≈ 0.20 + 0.80*0.25 = 0.40, NOT 0.55.
The ~15-point excess suggests acoustic features ALSO distinguish the mixed classes —
which would be a real, task-relevant shortcut.

This script isolates that by removing the benign original-vs-mixed distinction and asking
the question that actually matters for our component-level task:

  TEST A — MIXED-ONLY, 4-class:   drop `original`. Can acoustic-only features tell the four
           mixed classes apart? Chance = 1/4 = 0.25.
  TEST B — SPEECH spoof vs bonafide (binary, mixed only): among mixed clips, can acoustic
           features predict whether the SPEECH is spoofed? Chance = 0.50.
  TEST C — ENV spoof vs bonafide (binary, mixed only): same for the ENVIRONMENT component.

Filename features are DELIBERATELY EXCLUDED (the model never sees filenames).

Verdict logic (per test): accuracy near chance  -> that component does NOT leak acoustically
(safe). Accuracy well above chance -> real acoustic shortcut the benchmark construction must
control for (loudness/energy equalization, control C3).

Reuses the SAME acoustic feature extraction as audit_metadata_shortcuts.py for comparability.

Usage:
    python audit_acoustic_component_shortcut.py \
        --metadata /home/ee/btech/ee1230446/scratch/CompSpoofV2/metadata_normalized.csv \
        --data_root /home/ee/btech/ee1230446/scratch/CompSpoofV2 \
        --sample_per_class 800 --num_workers 8
"""
import argparse, os
import numpy as np
import pandas as pd
import soundfile as sf
import pyloudnorm as pyln
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

SEED = 13
MIXED = ["bonafide_bonafide", "bonafide_spoof", "spoof_bonafide", "spoof_spoof"]

# ACOUSTIC features only (no filename) — same set his script computes, minus fname/file_size
ACOUSTIC_COLS = ["duration_sec", "sample_peak", "rms", "loudness_lufs",
                 "mean_val", "std_val", "zcr"]


def extract_acoustic(args_tuple):
    """Same acoustic extraction as the original audit, for comparability."""
    rel_path, data_root, label = args_tuple
    full = os.path.join(data_root, rel_path)
    if not os.path.exists(full):
        return None
    try:
        data, sr = sf.read(full, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data**2)))
        meter = pyln.Meter(sr)
        try:
            loud = float(meter.integrated_loudness(data))
            if not np.isfinite(loud):
                loud = -70.0
        except Exception:
            loud = -70.0
        return {
            "label": label,
            "duration_sec": len(data) / float(sr),
            "sample_peak": peak,
            "rms": rms,
            "loudness_lufs": loud,
            "mean_val": float(np.mean(data)),
            "std_val": float(np.std(data)),
            "zcr": float(np.mean((data[:-1] * data[1:]) < 0)),
        }
    except Exception:
        return None


def evaluate(X, y, chance, title, out_lines):
    """5-fold CV of RF + LogReg (acoustic only), reported vs chance."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    Xs = StandardScaler().fit_transform(X)
    dummy = cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=skf).mean()
    rf = cross_val_score(RandomForestClassifier(n_estimators=200, max_depth=8,
                         random_state=SEED, n_jobs=-1), X, y, cv=skf, n_jobs=-1).mean()
    lr = cross_val_score(LogisticRegression(max_iter=1000), Xs, y, cv=skf).mean()
    best = max(rf, lr)
    margin = best - chance
    verdict = ("SAFE — acoustic features are near chance; this does NOT leak."
               if margin < 0.10 else
               "LEAK — acoustic features predict this well above chance. "
               "Construction must control for it (loudness/energy equalization).")
    out_lines += [
        "-" * 60,
        title,
        f"  n={len(y)}   classes={sorted(pd.Series(y).unique())}",
        f"  chance (even)        : {chance:.3f}",
        f"  majority baseline    : {dummy:.3f}",
        f"  Random Forest (5-CV) : {rf:.3f}",
        f"  LogReg (5-CV)        : {lr:.3f}",
        f"  --> best {best:.3f}  (margin over chance {margin:+.3f})",
        f"  VERDICT: {verdict}",
    ]
    return best, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sample_per_class", type=int, default=800)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default="acoustic_component_audit_report.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    df = df[df["split"].isin(["train", "val"])]
    df = df[df["label"].isin(MIXED)].copy()          # <-- drop `original`: the whole point

    # balanced sample across the 4 mixed classes
    parts = []
    for cls in MIXED:
        c = df[df["label"] == cls]
        parts.append(c.sample(n=min(len(c), args.sample_per_class), random_state=SEED))
    sample = pd.concat(parts, ignore_index=True)
    print(f"Sampled {len(sample)} MIXED clips ({args.sample_per_class}/class target).")

    tasks = [(r["audio_path"], args.data_root, r["label"]) for _, r in sample.iterrows()]
    rows = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for res in tqdm(ex.map(extract_acoustic, tasks), total=len(tasks)):
            if res:
                rows.append(res)
    feat = pd.DataFrame(rows)
    feat[ACOUSTIC_COLS] = feat[ACOUSTIC_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"Extracted acoustic features for {len(feat)} clips.\n")

    X = feat[ACOUSTIC_COLS]
    out = ["=" * 60,
           "ACOUSTIC-ONLY COMPONENT SHORTCUT AUDIT  (mixed clips only)",
           "Filename features excluded (model never sees them).",
           "=" * 60]

    # TEST A: 4-class among mixed
    evaluate(X, feat["label"], 0.25,
             "TEST A — 4 mixed classes (chance 0.25)", out)

    # TEST B: speech spoof vs bonafide  (label starts with 'spoof_' vs 'bonafide_')
    speech = feat["label"].str.split("_").str[0]      # bonafide / spoof (speech component)
    evaluate(X, speech, 0.50,
             "TEST B — SPEECH: spoof vs bonafide (chance 0.50)", out)

    # TEST C: env spoof vs bonafide  (second token)
    env = feat["label"].str.split("_").str[1]         # bonafide / spoof (env component)
    evaluate(X, env, 0.50,
             "TEST C — ENVIRONMENT: spoof vs bonafide (chance 0.50)", out)

    out += ["=" * 60,
            "READING THE RESULT:",
            " - If TEST A ~0.25 and B,C ~0.50 -> source data has NO acoustic component-level",
            "   shortcut; the earlier 55% was indeed just original-vs-mixed. Safe.",
            " - If any is well above chance -> a REAL acoustic shortcut exists for that",
            "   component; CompSpoof-TL construction must equalize it (control C3), and the",
            "   later C4 probe on the constructed benchmark must confirm it's removed.",
            "=" * 60]
    report = "\n".join(out)
    print("\n" + report)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()