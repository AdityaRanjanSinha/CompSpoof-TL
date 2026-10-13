"""
scripts/bootstrap_ci.py  --  95% bootstrap confidence intervals over clips (plan: "statistical
comparisons between systems use bootstrap confidence intervals over clips").

For one run, or a paired comparison of two runs scored on the same clips (A - B):
  pooled       frame EER @ 20 ms over all eval clips, per component
  seen/unseen  frame EER inside clips where that component was manipulated, split by whether
               its generator is unseen (SoulX-Podcast / MeanAudio)
  insert       spoofed-insert vs bonafide-insert (C1 twin) EER, per component

Clips are resampled with replacement (default 1000 times); CI = 2.5th / 97.5th percentiles.

Usage:
  PYTHONPATH=. python scripts/bootstrap_ci.py --manifest <eval/manifest.csv> --run_a <dir> [--run_b <dir>] [--n 1000]
"""
import argparse
import json
import numpy as np
import pandas as pd
from src.data.dataset import regions_to_mask
from src.metrics.frame_metrics import eer

UNSEEN = {"speech": "SoulX-Podcast", "env": "MeanAudio"}
GEN = {"speech": "speech_generator", "env": "env_generator"}
REG = {"speech": "speech_regions", "env": "env_regions"}


def load(run, ids_ref=None):
    z = dict(np.load(f"{run}/scores_eval.npz"))
    ids = list(z["clip_id"])
    if ids_ref is not None:
        pos = {c: i for i, c in enumerate(ids)}
        order = [pos[c] for c in ids_ref]
        z = {k: (v[order] if k != "clip_id" else np.array(ids_ref)) for k, v in z.items()}
    return z


def groups(z, man):
    """per component: clip index sets and the frame-label arrays each metric uses."""
    c1 = man["c1_matched_bonafide"].astype(str).str.lower().eq("true").values
    out = {}
    for c in ("speech", "env"):
        m = z[f"{c}_mask"].astype(int)
        manip = m.any(1)
        gens = np.array([str(g) for g in man[GEN[c]]])
        twin = np.stack([regions_to_mask(v) for v in man[REG[c]]]).astype(int) * c1[:, None]
        out[c] = {"mask": m, "manip": manip, "unseen": manip & (gens == UNSEEN[c]),
                  "seen": manip & (gens != UNSEEN[c]), "twin": twin}
    return out


def metrics(z, g, idx):
    r = {}
    for c in ("speech", "env"):
        s, m, G = z[f"{c}_scores"][idx], g[c]["mask"][idx], g[c]
        r[f"{c}_pooled"] = eer(s.ravel(), m.ravel())[0]
        for k in ("seen", "unseen"):
            sel = G[k][idx]
            r[f"{c}_{k}"] = eer(s[sel].ravel(), m[sel].ravel())[0] if sel.any() else np.nan
        pos = s[m == 1]; neg = s[G["twin"][idx] == 1]
        r[f"{c}_insert"] = eer(np.r_[pos, neg], np.r_[np.ones(len(pos)), np.zeros(len(neg))])[0] if len(neg) else np.nan
    return r


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--manifest", required=True)
    a.add_argument("--run_a", required=True)
    a.add_argument("--run_b", default=None)
    a.add_argument("--n", type=int, default=1000)
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()
    za = load(args.run_a)
    ids = list(za["clip_id"])
    zb = load(args.run_b, ids) if args.run_b else None
    man = pd.read_csv(args.manifest, dtype={"clip_id": str}, low_memory=False).set_index("clip_id").loc[ids]
    g = groups(za, man)
    full = np.arange(len(ids))
    pa = metrics(za, g, full); pb = metrics(zb, g, full) if zb else None
    rng = np.random.default_rng(args.seed)
    boots = {k: [] for k in pa}
    for i in range(args.n):
        idx = rng.integers(0, len(ids), len(ids))
        ra = metrics(za, g, idx)
        rb = metrics(zb, g, idx) if zb else None
        for k in pa:
            boots[k].append(ra[k] - rb[k] if zb else ra[k])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n}", flush=True)
    res = {}
    label = "A - B" if zb else "A"
    print(f"\n{'metric':16s} {'A':>8s} {'B':>8s}   {label} [95% CI]")
    for k in pa:
        v = np.array(boots[k], float); v = v[~np.isnan(v)]
        lo, hi = (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)
        point = pa[k] - pb[k] if zb else pa[k]
        res[k] = {"A": pa[k], "B": pb[k] if zb else None, "point": point, "ci95": [lo, hi]}
        print(f"{k:16s} {pa[k]:8.4f} {(pb[k] if zb else float('nan')):8.4f}   {point:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    out = f"{args.run_a}/bootstrap_{'vs_' + args.run_b.rstrip('/').split('/')[-1] if zb else 'single'}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
