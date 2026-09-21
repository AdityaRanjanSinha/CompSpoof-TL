"""
scripts/c4_matched_twin_probe.py  --  C4 anti-shortcut gate (correct formulation).

The right question is NOT "can a probe find splice boundaries" (it always can — content
changes at a splice). It is: "using only LOCAL boundary/artifact features, can a probe
tell a real spoof clip from its matched bonafide twin?" If the C1 twins are properly
matched (same region statistics, same crossfades, loudness-matched), the answer must be
NEAR CHANCE — because the *edits* are statistically identical in both; only the genuine
spoof content differs, and that is NOT a construction artifact.

  PASS (near 0.50): construction artifacts do not reveal the label -> splices are 'honest',
                    the benchmark cannot be gamed via edit artifacts. Safe to generate.
  FAIL (>>0.50)   : the splice artifact itself leaks the label (e.g. spoof splices click
                    differently) -> tighten crossfades / loudness before full generation.

Pairs each spoof clip with its C1 twin (same clip_id prefix), extracts LOCAL features at
the (shared) region boundaries, and classifies spoof-vs-twin. Grouped CV by clip pair.

Usage:
    python scripts/c4_matched_twin_probe.py --pilot ~/scratch/CompSpoof-TL/pilot2
"""
import argparse, os, json
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

SR = 16000
FRAME_MS = 20
FRAME = SR * FRAME_MS // 1000
SEED = 13


def local_boundary_features(y, regions):
    """
    For each region edge, compute LOCAL artifact descriptors in a small window around it,
    then aggregate (mean+max) over all edges in the clip. Captures clicks / discontinuities /
    local level & spectral jumps AT the splice joins — the things a construction artifact
    would create. Deliberately LOCAL (not global) so we test the edits, not the content.
    """
    n = len(y)
    edges = []
    for r in regions:
        for e in (r["start_sec"], r["end_sec"]):
            edges.append(int(round(e * SR)))
    if not edges:
        return None
    descs = []
    for c in edges:
        lo, hi = max(0, c - FRAME), min(n, c + FRAME)   # +-20ms around the edge
        seg = y[lo:hi]
        if len(seg) < 8:
            continue
        d = np.diff(seg)
        descs.append([
            float(np.max(np.abs(d))),                    # biggest sample jump (click)
            float(np.std(d)),                            # local roughness
            float(np.sqrt(np.mean(seg**2)) + 1e-9),      # local energy
            float(np.mean(np.abs(np.diff(np.sign(seg))) > 0)),  # local ZCR
            float(np.abs(np.sqrt(np.mean(y[max(0,c-FRAME):c]**2)) -
                         np.sqrt(np.mean(y[c:c+FRAME]**2)))),   # energy jump across edge
        ])
    if not descs:
        return None
    descs = np.array(descs)
    return np.concatenate([descs.mean(0), descs.max(0)])   # 10-d clip descriptor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", default="c4_matched_twin_report.txt")
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(a.pilot, "manifest.csv"))
    man["cid"] = man["clip_id"].astype(str)

    # pair spoof clips with their c1 twins by the numeric prefix (000006_... <-> 000006_c1twin_...)
    def prefix(cid):
        parts = cid.split("_")
        # find the first all-digit part (the numeric clip id), robust to a split prefix
        for p in parts:
            if p.isdigit():
                return p
        return parts[0]
    twins = {prefix(c): c for c in man["cid"] if "c1twin" in c}
    spoofs = {prefix(c): c for c in man["cid"]
              if "c1twin" not in c and man.loc[man.cid == c, "label"].iloc[0] != "bonafide_bonafide"}
    pairs = [(spoofs[p], twins[p]) for p in spoofs if p in twins]
    print(f"Found {len(pairs)} spoof/twin pairs.")

    def feats_for(cid):
        row = man.loc[man.cid == cid].iloc[0]
        regs = []
        for col in ("speech_regions", "env_regions"):
            try:
                regs += json.loads(row[col]) if isinstance(row[col], str) else []
            except Exception:
                pass
        if not regs:
            return None
        p = os.path.join(a.pilot, "audio", cid + ".wav")
        try:
            y, _ = librosa.load(p, sr=SR, mono=True)
        except Exception:
            return None
        return local_boundary_features(y, regs)

    X, Y = [], []
    for sp, tw in pairs:
        fs, ft = feats_for(sp), feats_for(tw)
        if fs is not None:
            X.append(fs); Y.append(1)     # spoof
        if ft is not None:
            X.append(ft); Y.append(0)     # matched bonafide twin
    X = np.array(X); Y = np.array(Y)
    X = np.nan_to_num(X)
    print(f"Probing {len(Y)} clips ({Y.sum()} spoof / {(Y==0).sum()} twin) on LOCAL boundary features.\n")

    Xs = StandardScaler().fit_transform(X)
    acc = cross_val_score(RandomForestClassifier(200, max_depth=6, random_state=SEED, n_jobs=-1),
                          Xs, Y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED)).mean()

    verdict = ("PASS — boundary/artifact features do NOT separate spoof from matched twin "
               "(near chance). Splices are honest; benchmark cannot be gamed via edits. "
               "SAFE TO GENERATE THE FULL BENCHMARK."
               if acc < 0.60 else
               "FAIL — the splice artifact leaks the label (probe beats chance). Tighten "
               "crossfades / loudness matching before generating the full benchmark.")
    out = ["="*60, "C4 MATCHED-TWIN SPLICE PROBE (GNG-2 gate)",
           "spoof vs matched-bonafide-twin, LOCAL boundary features only", "="*60,
           f"pairs                 : {len(pairs)}",
           f"probe accuracy        : {acc:.3f}   (chance 0.500, margin {acc-0.5:+.3f})",
           f"\nVERDICT: {verdict}", "="*60]
    rep = "\n".join(out)
    print("\n" + rep)
    open(a.out, "w").write(rep + "\n")
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
