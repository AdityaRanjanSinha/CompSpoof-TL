"""
scripts/diagnose_env_leak.py  --  diagnose WHAT spectral property carries the residual
environment acoustic shortcut in the generated pilot.

Context: C4 audit showed env spoof-vs-bonafide still leaks ~0.61 (chance 0.50) after
loudness matching (C3). Before building any fix, we need to know WHICH features leak, so
the fix targets the right thing (spectral envelope? high-freq detail? MFCCs?).

Method: on the generated pilot's ENV component (we reconstruct env-only from the manifest
by using the env source, OR fall back to the mixed clip), extract a RICH feature set:
  - level:    rms, peak            (should already be matched by C3)
  - spectral: centroid, bandwidth, rolloff, flatness, band-energy ratios (low/mid/high)
  - detail:   MFCCs (1-13) mean + std   (captures fine spectral / vocoder character)
Then train a classifier per feature GROUP and report which group leaks, plus per-feature
importances. Whatever leaks tells us what the fix must equalize.

Usage:
    python scripts/diagnose_env_leak.py --pilot ~/scratch/CompSpoof-TL/pilot2
"""
import argparse, os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

SR = 16000
SEED = 13

LEVEL   = ["rms", "peak"]
SPECTRAL= ["centroid", "bandwidth", "rolloff", "flatness",
           "band_low", "band_mid", "band_high"]
MFCC    = [f"mfcc{i}_mean" for i in range(1, 14)] + [f"mfcc{i}_std" for i in range(1, 14)]


def features(y):
    eps = 1e-9
    f = {}
    f["rms"] = float(np.sqrt(np.mean(y**2) + eps))
    f["peak"] = float(np.max(np.abs(y)) + eps)
    S = np.abs(librosa.stft(y, n_fft=1024)) + eps
    f["centroid"] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=SR)))
    f["bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=SR)))
    f["rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=SR)))
    f["flatness"] = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    # band-energy ratios (low <1k, mid 1-4k, high >4k)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    P = (S**2).mean(axis=1); P = P / (P.sum() + eps)
    f["band_low"] = float(P[freqs < 1000].sum())
    f["band_mid"] = float(P[(freqs >= 1000) & (freqs < 4000)].sum())
    f["band_high"] = float(P[freqs >= 4000].sum())
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    for i in range(13):
        f[f"mfcc{i+1}_mean"] = float(mf[i].mean())
        f[f"mfcc{i+1}_std"] = float(mf[i].std())
    return f


def evaluate(X, y, cols, name, out):
    if not set(cols).issubset(X.columns):
        return
    Xg = StandardScaler().fit_transform(X[cols])
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    acc = cross_val_score(RandomForestClassifier(200, max_depth=8, random_state=SEED,
                          n_jobs=-1), Xg, y, cv=skf, n_jobs=-1).mean()
    out.append(f"  {name:16s}: {acc:.3f}  (chance 0.500, margin {acc-0.5:+.3f})")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", default="env_leak_diagnosis.txt")
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(a.pilot, "manifest.csv"))
    man = man[~man["clip_id"].astype(str).str.contains("c1twin")]
    # env-spoof label: from the class name (2nd token) — spoof vs bonafide env
    man = man[man["label"].str.contains("_")].copy()
    man["env_is"] = man["label"].str.split("_").str[1]   # bonafide / spoof

    rows, labels = [], []
    for _, r in man.iterrows():
        p = os.path.join(a.pilot, "audio", str(r["clip_id"]) + ".wav")
        try:
            y, _ = librosa.load(p, sr=SR, mono=True)
            rows.append(features(y)); labels.append(r["env_is"])
        except Exception:
            pass
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.array([1 if l == "spoof" else 0 for l in labels])
    print(f"Diagnosing {len(X)} clips on env spoof-vs-bonafide.\n")

    out = ["="*60, "ENV LEAK DIAGNOSIS — which feature group carries the shortcut?",
           "(env spoof vs bonafide; chance = 0.500)", "="*60]
    evaluate(X, y, LEVEL, "level (rms/peak)", out)
    evaluate(X, y, SPECTRAL, "spectral shape", out)
    evaluate(X, y, MFCC, "MFCC (detail)", out)
    evaluate(X, y, LEVEL + SPECTRAL + MFCC, "ALL", out)

    # top leaking individual features
    clf = RandomForestClassifier(300, random_state=SEED, n_jobs=-1)
    clf.fit(StandardScaler().fit_transform(X), y)
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    out += ["", "Top 12 leaking features:", imp.head(12).to_string(),
            "", "READING:",
            " - If LEVEL leaks -> C3 loudness matching is not tight enough.",
            " - If SPECTRAL leaks -> need spectral-envelope matching (equalize band ratios).",
            " - If MFCC leaks most -> fine spectral/vocoder character; envelope matching may",
            "   only partially help -> consider documenting residual as a known limitation.",
            "="*60]
    rep = "\n".join(out)
    print("\n" + rep)
    open(a.out, "w").write(rep + "\n")
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
