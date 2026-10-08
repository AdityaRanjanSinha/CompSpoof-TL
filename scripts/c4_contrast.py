"""
scripts/c4_contrast.py  --  does a model separate SPOOFED inserts from BONAFIDE inserts?

The C4 probe (trained only on all-bonafide splices) localizes spoofs well, so splicing itself
is detectable. The question that matters is whether a model can tell a spoofed inserted region
from a bonafide inserted region (the C1 twins). Per component:

  positives = frames inside spoofed regions (spoof masks, spoofed clips)
  negatives = frames inside the splice regions of C1 bonafide twins (manifest *_regions)

EER near 0.5 here = the model only detects "something was inserted", not spoofing.
Also reports a boundary profile: mean score inside spoofed regions vs distance from the
nearest region edge (0-2, 3-5, 6-10, >10 frames) and outside regions.

Usage: PYTHONPATH=. python scripts/c4_contrast.py --manifest <eval/manifest.csv> --run_dir A --run_dir B ...
"""
import argparse
import numpy as np
import pandas as pd
from src.data.dataset import regions_to_mask
from src.metrics.frame_metrics import eer


def edge_distance(mask):
    """per-frame distance (in frames) to the nearest region edge, for frames inside the mask."""
    m = mask.astype(bool)
    d = np.full(len(m), -1)
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]) != 0)   # region starts / ends
    if len(edges) == 0:
        return d
    idx = np.arange(len(m))
    dist = np.min(np.abs(idx[:, None] - edges[None, :]), axis=1)
    d[m] = dist[m]
    return d


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--manifest", required=True)
    a.add_argument("--run_dir", action="append", required=True)
    args = a.parse_args()
    df = pd.read_csv(args.manifest, dtype={"clip_id": str}, low_memory=False).set_index("clip_id")
    for run in args.run_dir:
        z = dict(np.load(f"{run}/scores_eval.npz"))
        ids = list(z["clip_id"])
        man = df.loc[ids]
        c1 = man["c1_matched_bonafide"].astype(str).str.lower().eq("true").values
        print(f"\n== {run}   ({c1.sum()} C1 twins of {len(ids)} clips)")
        for c, col in (("speech", "speech_regions"), ("env", "env_regions")):
            s, m = z[f"{c}_scores"], z[f"{c}_mask"]
            spoofed = m.any(1)
            twin = np.stack([regions_to_mask(v) for v in man[col]]) * c1[:, None]
            pos = s[m == 1]
            neg = s[(twin == 1)]
            e = eer(np.r_[pos, neg], np.r_[np.ones(len(pos)), np.zeros(len(neg))])[0] if len(neg) else float("nan")
            # boundary profile inside spoofed regions
            prof = {k: [] for k in ("0-2", "3-5", "6-10", ">10")}
            for i in np.flatnonzero(spoofed):
                d = edge_distance(m[i])
                for k, (lo, hi) in {"0-2": (0, 2), "3-5": (3, 5), "6-10": (6, 10), ">10": (11, 999)}.items():
                    sel = (d >= lo) & (d <= hi)
                    prof[k] += list(s[i][sel])
            outside = s[spoofed][m[spoofed] == 0].mean()
            prof_s = "  ".join(f"{k}:{np.mean(v):.2f}" for k, v in prof.items() if v)
            print(f"  {c:6s} spoof-region vs bonafide-splice-region EER {e:.3f}  "
                  f"(pos {len(pos)}, neg {len(neg)} frames)")
            print(f"  {'':6s} mean score in spoofed regions by edge distance [frames]: {prof_s}  | outside: {outside:.2f}")


if __name__ == "__main__":
    main()
