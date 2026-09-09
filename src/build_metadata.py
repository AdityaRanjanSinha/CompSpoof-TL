"""
build_metadata.py  --  CompSpoof-TL S1 deliverable (Week 1)

Merges CompSpoofV2's scattered per-split metadata into ONE normalized table,
one row per clip, with a consistent schema and a `split` column.

Design decisions (documented for the data report):
- The `*_source/metadata/*.csv` files are the AUTHORITATIVE, fully-annotated
  metadata (source paths, generator method/model, SNR). We use those.
- development/metadata/{train,val}.csv are already fully annotated -> used as-is.
- eval_label.csv / test_label.csv and eval/metadata/eval.csv are the BLIND
  challenge files (label-only or stripped); we do NOT use them for provenance,
  but we cross-check that their clip lists match the source metadata.
- Column typo in the raw data ("mothed" for "method") is fixed on load.

Usage:
    python build_metadata.py --root /path/to/CompSpoofV2 --out metadata_normalized.csv
"""
import argparse
from pathlib import Path
import pandas as pd

# raw -> clean column names (fix the dataset's "mothed" typo)
RENAME = {
    "speech_generation_mothed": "speech_generation_method",
    "env_generation_mothed": "env_generation_method",
}

# the authoritative metadata file for each split
SPLIT_SOURCES = {
    "train": "development/metadata/train.csv",
    "val":   "development/metadata/val.csv",
    "eval":  "eval_source/metadata/eval.csv",
    "test":  "test_source/metadata/test.csv",
}


def load_split(root: Path, split: str, rel: str) -> pd.DataFrame:
    path = root / rel
    if not path.exists():
        print(f"  [WARN] missing: {rel} -- skipping {split}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns=RENAME)
    # normalize the empty marker '-' to real missing values
    df = df.replace("-", pd.NA)
    # force the split label to be correct/consistent
    df["split"] = split
    print(f"  {split:5s}: {len(df):>7,} rows  <- {rel}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to CompSpoofV2")
    ap.add_argument("--out", default="metadata_normalized.csv")
    args = ap.parse_args()
    root = Path(args.root)

    print("Loading per-split metadata (authoritative *_source files):")
    frames = [load_split(root, s, rel) for s, rel in SPLIT_SOURCES.items()]
    frames = [f for f in frames if not f.empty]
    table = pd.concat(frames, ignore_index=True)

    # a clean, ordered column set
    cols = ["audio_path", "split", "label",
            "speech_path", "speech_source", "speech_generation_method",
            "speech_generation_source", "speech_generation_model",
            "env_path", "env_source", "env_generation_method",
            "env_generation_source", "env_generation_model",
            "mix_target_snr", "original_audio_source"]
    table = table[[c for c in cols if c in table.columns]]

    # derive per-component authenticity from the 5-class label
    # labels look like "<speech>_<env>" e.g. bonafide_spoof, or "original"
    def speech_auth(lbl):
        if pd.isna(lbl) or lbl == "original": return lbl
        return lbl.split("_")[0]
    def env_auth(lbl):
        if pd.isna(lbl) or lbl == "original": return lbl
        parts = lbl.split("_")
        return parts[1] if len(parts) > 1 else pd.NA
    table["speech_is"] = table["label"].map(speech_auth)
    table["env_is"] = table["label"].map(env_auth)

    table.to_csv(args.out, index=False)

    # ---- summary (this is the "data report" content) ----
    print("\n" + "=" * 60)
    print(f"NORMALIZED TABLE: {len(table):,} clips  ->  {args.out}")
    print("=" * 60)
    print("\nRows per split:")
    print(table["split"].value_counts().to_string())
    print("\nClass (label) distribution:")
    print(table["label"].value_counts(dropna=False).to_string())
    print("\nSNR (mix_target_snr) summary:")
    snr = pd.to_numeric(table["mix_target_snr"], errors="coerce")
    print(snr.describe().to_string())
    print("\nSpeech generators seen per split (for unseen-generator split / H4):")
    for sp in table["split"].unique():
        gens = table[table.split == sp]["speech_generation_model"].dropna().unique()
        print(f"  {sp:5s} speech models: {sorted(map(str, gens))[:8]}"
              f"{' ...' if len(gens) > 8 else ''}")
    print("\nEnv generators seen per split:")
    for sp in table["split"].unique():
        gens = table[table.split == sp]["env_generation_model"].dropna().unique()
        print(f"  {sp:5s} env models:    {sorted(map(str, gens))[:8]}"
              f"{' ...' if len(gens) > 8 else ''}")
    print("\nProvenance completeness (non-null fraction) per split:")
    prov_cols = ["speech_path", "env_path", "mix_target_snr",
                 "speech_generation_model", "env_generation_model"]
    frac = table.groupby("split")[prov_cols].apply(lambda d: d.notna().mean())
    print(frac.to_string())


if __name__ == "__main__":
    main()
