"""
scripts/breakdown_eval.py  --  where does the model fail on eval?

Part A (clip groups): clips with no spoofed frames / spoofed clips with only seen generators /
spoofed clips flagged unseen_generator.
Part B (per component, per generator): for each component, the clips where THAT component was
manipulated, grouped by THAT component's generator. EER is computed inside those clips (spoofed
vs untouched frames of the same clips); recall = share of spoofed frames above the val-EER
threshold. This separates SoulX-Podcast (speech) and MeanAudio (env) from the seen generators.

Usage: PYTHONPATH=. python scripts/breakdown_eval.py --run_dir <run> --manifest <eval/manifest.csv>
"""
import argparse
import json
import numpy as np
import pandas as pd
from src.metrics.frame_metrics import eer

UNSEEN = {"speech": {"SoulX-Podcast"}, "env": {"MeanAudio"}}
GEN_COL = {"speech": "speech_generator", "env": "env_generator"}


def flag(v):
    return str(v).strip().lower() not in ("", "nan", "none", "false", "0")


def stats(s, m, thr):
    e = eer(s.ravel(), m.ravel())[0]
    rec = float((s[m == 1] >= thr).mean())
    fa = float((s[m == 0] >= thr).mean())
    return e, rec, fa


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run_dir", required=True)
    a.add_argument("--manifest", required=True)
    args = a.parse_args()
    z = dict(np.load(f"{args.run_dir}/scores_eval.npz"))     # load once (npz access is lazy)
    zv = dict(np.load(f"{args.run_dir}/scores_val.npz"))
    thr = {c: eer(zv[f"{c}_scores"].ravel(), zv[f"{c}_mask"].ravel())[1] for c in ("speech", "env")}
    df = pd.read_csv(args.manifest, dtype={"clip_id": str}, low_memory=False).set_index("clip_id")
    ids = list(z["clip_id"])
    man = df.loc[ids]
    out = {"val_thresholds": thr}
    print(f"val EER thresholds: speech {thr['speech']:.3f} env {thr['env']:.3f}")

    # ---- Part A: clip groups -------------------------------------------------------------
    print("\nA) clip groups   (false-alarm rate on untouched frames, at the val threshold)")
    any_sp = z["speech_mask"].any(1) | z["env_mask"].any(1)
    unseen = np.array([flag(u) for u in man["unseen_generator"]])
    groups = {"no spoof frames": ~any_sp, "spoofed, seen gens only": any_sp & ~unseen,
              "spoofed, has unseen gen": any_sp & unseen}
    out["groups"] = {}
    for g, sel in groups.items():
        line = f"  {g:26s} n={sel.sum():5d}"
        row = {"n": int(sel.sum())}
        for c in ("speech", "env"):
            s, m = z[f"{c}_scores"][sel], z[f"{c}_mask"][sel]
            fa = float((s[m == 0] >= thr[c]).mean())
            line += f" | {c} FA {fa:.3f}"
            row[f"{c}_false_alarm"] = fa
        print(line)
        out["groups"][g] = row

    # ---- Part B: per component, per generator --------------------------------------------
    out["per_generator"] = {}
    for c in ("speech", "env"):
        print(f"\nB) {c.upper()} -- clips where the {c} component was manipulated, by its generator")
        mm = z[f"{c}_mask"].any(1)
        gens = np.array([str(x) for x in man[GEN_COL[c]]])
        rows = []
        for gname in sorted(set(gens[mm])):
            sel = mm & (gens == gname)
            e, rec, fa = stats(z[f"{c}_scores"][sel], z[f"{c}_mask"][sel], thr[c])
            tag = "UNSEEN" if gname in UNSEEN[c] else "seen"
            rows.append((tag, gname, int(sel.sum()), e, rec, fa))
        for tag in ("seen", "UNSEEN"):
            sel = mm & np.array([(g in UNSEEN[c]) == (tag == "UNSEEN") for g in gens])
            if sel.sum():
                e, rec, fa = stats(z[f"{c}_scores"][sel], z[f"{c}_mask"][sel], thr[c])
                rows.append((tag, f"== ALL {tag} ==", int(sel.sum()), e, rec, fa))
        print(f"  {'':6s} {'generator':28s} {'clips':>6s} {'EER':>6s} {'recall':>7s} {'FA':>6s}")
        for tag, gname, n, e, rec, fa in rows:
            print(f"  {tag:6s} {gname[:28]:28s} {n:6d} {e:6.3f} {rec:7.3f} {fa:6.3f}")
        out["per_generator"][c] = [dict(zip(("tag", "generator", "clips", "eer", "recall", "false_alarm"), r))
                                   for r in rows]
    with open(f"{args.run_dir}/breakdown_eval.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
