"""
scripts/generate_benchmark.py  --  Week-5 S1: CompSpoof-TL pilot generation.

Runs the toolchain v1 orchestrator (construct_clip) on REAL CompSpoofV2 sources to
produce actual benchmark clips, each with:
  - a 4 s / 16 kHz mono mixture (.wav)
  - two per-component frame-level label masks (speech, env) at 20 ms  (.npy)
  - a manifest row (provenance: sources, regions, SNR, seed, label)  (manifest.csv)

How sources are chosen
----------------------
The normalized metadata gives, per existing clip, `speech_path`, `env_path`, and
`speech_is` / `env_is` (bonafide|spoof). We POOL these into four source lists:
  bonafide-speech, spoof-speech, bonafide-env, spoof-env.
To build a new clip of a target class we sample a bonafide *base* for each component,
plus a spoof source where that component is manipulated. SNR is sampled from the config
range (7-12 dB, from the Week-2 audit).

Class recipe (which components get a spoof source):
  bonafide_bonafide : none
  spoof_bonafide    : speech only
  bonafide_spoof    : env only
  spoof_spoof       : both
Plus, for anti-shortcut control C1, an optional matched bonafide twin per manipulated clip
(same seed -> same region statistics, spliced with a *different bonafide* source).

Scales from a handful of validation clips to the full 2,000-clip pilot by changing --n.

Usage:
    python scripts/generate_benchmark.py \
        --meta  /path/CompSpoofV2/metadata_normalized.csv \
        --root  /path/CompSpoofV2 \
        --out   /path/CompSpoof-TL/pilot \
        --split train --n 20 --seed 13 --with-c1
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

from src.toolchain.construct_clip import construct_clip, build_matched_bonafide

SR = 16000
CLIP_SEC = 4.0

# how many spoofed components each class needs
CLASS_RECIPE = {
    "bonafide_bonafide": (False, False),
    "spoof_bonafide":    (True,  False),
    "bonafide_spoof":    (False, True),
    "spoof_spoof":       (True,  True),
}


def load_wav(path: Path) -> np.ndarray:
    """Load a wav as 16 kHz mono float32, trimmed/padded to 4 s."""
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    n = int(CLIP_SEC * SR)
    out = np.zeros(n, dtype=np.float32)
    out[:min(n, len(y))] = y[:n]
    return out


def build_source_pools(df: pd.DataFrame):
    """Pool real source paths by component + authenticity."""
    pools = {
        "speech_bona": df.loc[df["speech_is"] == "bonafide", "speech_path"].dropna().unique().tolist(),
        "speech_spoof": df.loc[df["speech_is"] == "spoof", "speech_path"].dropna().unique().tolist(),
        "env_bona": df.loc[df["env_is"] == "bonafide", "env_path"].dropna().unique().tolist(),
        "env_spoof": df.loc[df["env_is"] == "spoof", "env_path"].dropna().unique().tolist(),
    }
    for k, v in pools.items():
        print(f"  pool {k:12s}: {len(v):>7,} sources")
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20, help="clips per class")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--snr_min", type=float, default=7.0)
    ap.add_argument("--snr_max", type=float, default=12.0)
    ap.add_argument("--with-c1", action="store_true", help="also emit C1 matched-bonafide twins")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out); (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.meta)
    df = df[df["split"] == args.split]
    pools = build_source_pools(df)
    region_cfg = dict(n_min=1, n_max=3, dur_min=0.4, dur_max=2.0)

    def pick(pool):
        return root / pools[pool][int(rng.integers(len(pools[pool])))]

    manifest = []
    clip_id = 0
    for cls, (sp_manip, en_manip) in CLASS_RECIPE.items():
        for _ in range(args.n):
            clip_id += 1
            seed = int(rng.integers(1_000_000_000))
            # bonafide bases (always)
            sp_base = load_wav(pick("speech_bona"))
            en_base = load_wav(pick("env_bona"))
            # spoof sources where manipulated
            sp_spoof = load_wav(pick("speech_spoof")) if sp_manip else None
            en_spoof = load_wav(pick("env_spoof")) if en_manip else None
            snr = float(rng.uniform(args.snr_min, args.snr_max))

            res = construct_clip(sp_base, en_base, snr_db=snr, seed=seed,
                                 speech_spoof=sp_spoof, env_spoof=en_spoof,
                                 region_cfg=region_cfg)

            name = f"{clip_id:06d}_{cls}"
            sf.write(out / "audio" / f"{name}.wav", res.audio, SR)
            np.save(out / "masks" / f"{name}_speech.npy", res.speech_mask)
            np.save(out / "masks" / f"{name}_env.npy", res.env_mask)
            row = {"clip_id": name, "label": cls, "split": args.split,
                   **res.provenance}
            row["speech_regions"] = json.dumps(row["speech_regions"])
            row["env_regions"] = json.dumps(row["env_regions"])
            manifest.append(row)

            # --- C1 matched bonafide twin (only for clips that had splices) ---
            if args.with_c1 and (sp_manip or en_manip):
                sp_alt = load_wav(pick("speech_bona")) if sp_manip else None
                en_alt = load_wav(pick("env_bona")) if en_manip else None
                twin = build_matched_bonafide(sp_base, en_base, sp_alt, en_alt,
                                              snr_db=snr, seed=seed, region_cfg=region_cfg)
                tname = f"{clip_id:06d}_c1twin_bonafide_bonafide"
                sf.write(out / "audio" / f"{tname}.wav", twin.audio, SR)
                np.save(out / "masks" / f"{tname}_speech.npy", twin.speech_mask)
                np.save(out / "masks" / f"{tname}_env.npy", twin.env_mask)
                trow = {"clip_id": tname, "label": "bonafide_bonafide", "split": args.split,
                        **twin.provenance}
                trow["speech_regions"] = json.dumps(trow["speech_regions"])
                trow["env_regions"] = json.dumps(trow["env_regions"])
                manifest.append(trow)

    man = pd.DataFrame(manifest)
    man.to_csv(out / "manifest.csv", index=False)

    print(f"\nGenerated {len(man)} clips -> {out}")
    print("Per-label counts:")
    print(man["label"].value_counts().to_string())
    print(f"\nMean speech-spoof frac: {man['speech_spoof_frac'].mean():.3f}   "
          f"mean env-spoof frac: {man['env_spoof_frac'].mean():.3f}")
    print(f"SNR range used: {man['snr_db'].min():.1f} - {man['snr_db'].max():.1f} dB")
    print(f"manifest -> {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
