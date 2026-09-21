"""
scripts/generate_benchmark_v2.py  --  CompSpoof-TL benchmark generation with proper splits.

Upgrades generate_benchmark.py to build the full train / val / eval structure the plan needs:

  train : spoof+bonafide sources pooled from the DEVELOPMENT split (train rows)
  val   : sources pooled from the DEVELOPMENT split (val rows)   [held-out sources]
  eval  : sources pooled from EVAL+TEST source rows -> inherits CompSpoofV2's UNSEEN
          generators (env MeanAudio, speech SoulX-Podcast appear only here). Each eval
          clip is TAGGED with whether its spoof source used an unseen generator, so the
          seen-vs-unseen generalization comparison (H4) is possible.

Each generated clip -> mixture .wav + speech/env frame-mask .npy + manifest row.
C1 matched-bonafide twins optional (--with-c1). Reuses the v1 orchestrator (construct_clip).

Anti-shortcut status (verified on pilot): C3 loudness matching + C1 twins pass the C4
splice-probe gate (0.34 << 0.5). MFCC residual is intrinsic generated-audio character
(documented, not scrubbed). Safe to generate.

Usage (build all three splits):
    python scripts/generate_benchmark_v2.py --meta META --root ROOT \
        --out ~/scratch/CompSpoof-TL/full --split train --n 13750 --seed 13 --with-c1
    python scripts/generate_benchmark_v2.py ... --split val  --n 1750 --seed 42
    python scripts/generate_benchmark_v2.py ... --split eval --n 3400 --seed 2027
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

from src.toolchain.construct_clip import construct_clip, build_matched_bonafide

SR = 16000
CLIP_SEC = 4.0
CLASS_RECIPE = {
    "bonafide_bonafide": (False, False),
    "spoof_bonafide":    (True,  False),
    "bonafide_spoof":    (False, True),
    "spoof_spoof":       (True,  True),
}
# generators that appear ONLY in eval/test (unseen at train) -> from Week-2 metadata findings
UNSEEN_SPEECH_GENS = {"SoulX-Podcast"}
UNSEEN_ENV_GENS = {"MeanAudio"}

# which metadata splits feed each benchmark split
SPLIT_SOURCE_ROWS = {
    "train": ["train"],
    "val":   ["val"],
    "eval":  ["eval", "test"],     # eval draws from held-out source rows
}


def load_wav(path):
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    n = int(CLIP_SEC * SR)
    out = np.zeros(n, dtype=np.float32)
    out[:min(n, len(y))] = y[:n]
    return out


def build_pools(df):
    """Pool (path, generation_model) tuples by component + authenticity."""
    def pool(mask, path_col, gen_col):
        sub = df.loc[mask, [path_col, gen_col]].dropna(subset=[path_col])
        return list(sub.itertuples(index=False, name=None))   # (path, gen_model)
    pools = {
        "speech_bona":  pool(df["speech_is"] == "bonafide", "speech_path", "speech_generation_model"),
        "speech_spoof": pool(df["speech_is"] == "spoof",    "speech_path", "speech_generation_model"),
        "env_bona":     pool(df["env_is"] == "bonafide",    "env_path",    "env_generation_model"),
        "env_spoof":    pool(df["env_is"] == "spoof",       "env_path",    "env_generation_model"),
    }
    for k, v in pools.items():
        print(f"  pool {k:12s}: {len(v):>7,}")
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "eval"])
    ap.add_argument("--n", type=int, default=100, help="clips per class")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--snr_min", type=float, default=7.0)
    ap.add_argument("--snr_max", type=float, default=12.0)
    ap.add_argument("--with-c1", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) / args.split
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.meta)
    src_rows = SPLIT_SOURCE_ROWS[args.split]
    df = df[df["split"].isin(src_rows)]
    print(f"[{args.split}] pooling sources from metadata splits {src_rows}:")
    pools = build_pools(df)
    region_cfg = dict(n_min=1, n_max=3, dur_min=0.4, dur_max=2.0)

    def pick(pool):
        p = pools[pool]
        path, gen = p[int(rng.integers(len(p)))]
        return root / path, gen

    manifest = []
    cid = 0
    for cls, (sp_manip, en_manip) in CLASS_RECIPE.items():
        for _ in range(args.n):
            cid += 1
            seed = int(rng.integers(1_000_000_000))
            sp_path, _ = pick("speech_bona"); sp_base = load_wav(sp_path)
            en_path, _ = pick("env_bona");    en_base = load_wav(en_path)
            sp_spoof = en_spoof = None
            sp_gen = en_gen = None
            if sp_manip:
                p, sp_gen = pick("speech_spoof"); sp_spoof = load_wav(p)
            if en_manip:
                p, en_gen = pick("env_spoof");    en_spoof = load_wav(p)
            snr = float(rng.uniform(args.snr_min, args.snr_max))

            res = construct_clip(sp_base, en_base, snr_db=snr, seed=seed,
                                 speech_spoof=sp_spoof, env_spoof=en_spoof,
                                 region_cfg=region_cfg)
            unseen = (str(sp_gen) in UNSEEN_SPEECH_GENS) or (str(en_gen) in UNSEEN_ENV_GENS)
            name = f"{args.split}_{cid:06d}_{cls}"
            sf.write(out / "audio" / f"{name}.wav", res.audio, SR)
            np.save(out / "masks" / f"{name}_speech.npy", res.speech_mask)
            np.save(out / "masks" / f"{name}_env.npy", res.env_mask)
            row = {"clip_id": name, "label": cls, "split": args.split,
                   "speech_generator": sp_gen, "env_generator": en_gen,
                   "unseen_generator": bool(unseen), **res.provenance}
            row["speech_regions"] = json.dumps(row["speech_regions"])
            row["env_regions"] = json.dumps(row["env_regions"])
            manifest.append(row)

            if args.with_c1 and (sp_manip or en_manip):
                sp_alt = pick("speech_bona")[0] if sp_manip else None
                en_alt = pick("env_bona")[0] if en_manip else None
                sp_alt = load_wav(sp_alt) if sp_alt is not None else None
                en_alt = load_wav(en_alt) if en_alt is not None else None
                twin = build_matched_bonafide(sp_base, en_base, sp_alt, en_alt,
                                              snr_db=snr, seed=seed, region_cfg=region_cfg)
                tname = f"{args.split}_{cid:06d}_c1twin_bonafide_bonafide"
                sf.write(out / "audio" / f"{tname}.wav", twin.audio, SR)
                np.save(out / "masks" / f"{tname}_speech.npy", twin.speech_mask)
                np.save(out / "masks" / f"{tname}_env.npy", twin.env_mask)
                trow = {"clip_id": tname, "label": "bonafide_bonafide", "split": args.split,
                        "speech_generator": None, "env_generator": None,
                        "unseen_generator": False, **twin.provenance}
                trow["speech_regions"] = json.dumps(trow["speech_regions"])
                trow["env_regions"] = json.dumps(trow["env_regions"])
                manifest.append(trow)

    man = pd.DataFrame(manifest)
    man.to_csv(out / "manifest.csv", index=False)
    print(f"\n[{args.split}] generated {len(man)} clips -> {out}")
    print(man["label"].value_counts().to_string())
    if args.split == "eval":
        n_unseen = int(man["unseen_generator"].sum())
        print(f"unseen-generator clips: {n_unseen} / {len(man)} "
              f"(for the seen-vs-unseen H4 comparison)")
    print(f"SNR: {man['snr_db'].min():.1f}-{man['snr_db'].max():.1f} dB   manifest -> {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
