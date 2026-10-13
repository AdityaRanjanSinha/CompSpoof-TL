"""
scripts/eval_scores_npz.py  --  full CompSpoof-TL metric suite on any saved scores_<split>.npz
(e.g. the ESDD2 sliding-window baseline), split into all / seen / unseen generators,
with the same numbers train.py reports for our model.

Usage: PYTHONPATH=. python scripts/eval_scores_npz.py --run_dir <dir> --manifest <split>/manifest.csv [--split eval]
"""
import argparse
import json
import numpy as np
import pandas as pd
from src.metrics.evaluate import evaluate_suite


def flag(v):
    return str(v).strip().lower() not in ("", "nan", "none", "false", "0")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run_dir", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--split", default="eval")
    args = a.parse_args()
    z = dict(np.load(f"{args.run_dir}/scores_{args.split}.npz"))
    ids = list(z["clip_id"])
    sp_s, en_s = list(z["speech_scores"]), list(z["env_scores"])
    sp_m, en_m = list(z["speech_mask"].astype(int)), list(z["env_mask"].astype(int))
    res = {"all": evaluate_suite(sp_s, sp_m, en_s, en_m), "n_clips": len(ids)}
    man = pd.read_csv(args.manifest, dtype={"clip_id": str}, low_memory=False).set_index("clip_id")
    if "unseen_generator" in man.columns:
        unseen = np.array([flag(v) for v in man.loc[ids, "unseen_generator"]])
        for name, sel in (("seen", ~unseen), ("unseen", unseen)):
            idx = np.flatnonzero(sel)
            if len(idx):
                res[name] = evaluate_suite([sp_s[i] for i in idx], [sp_m[i] for i in idx],
                                           [en_s[i] for i in idx], [en_m[i] for i in idx])
                res[f"n_{name}"] = int(len(idx))
    with open(f"{args.run_dir}/results_{args.split}.json", "w") as f:
        json.dump(res, f, indent=2)
    for k in ("all", "seen", "unseen"):
        if k in res:
            r = res[k]
            print(f"[{k:6s}] frame-EER@20ms speech {r['speech']['frame_eer'][20]:.4f} env {r['environment']['frame_eer'][20]:.4f} "
                  f"| event-F1 {r['speech']['event_f1']:.3f}/{r['environment']['event_f1']:.3f} "
                  f"| clip-correct {r['joint']['clip_correct']:.3f} | macro-F1 {r['utterance']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
