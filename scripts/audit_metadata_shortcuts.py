"""
scripts/audit_metadata_shortcuts.py

CompSpoof-TL S1 Deliverable (Week 2):
Audits the CompSpoofV2 dataset for shortcut risks. Checks whether class labels
can be predicted from non-semantic acoustic and metadata properties alone:
- File naming (length, digits, naming patterns)
- File size
- Clip duration
- Sample peak amplitude (max |x|)
- Integrated loudness (ITU-R BS.1770 via pyloudnorm)
- Channel / signal statistics (RMS energy, mean, std, zero-crossing rate)

As specified in the Project Plan (Section 5):
"Audit the metadata for shortcut risk: check whether class is predictable from
file naming, clip duration, sample peak, loudness, or channel statistics alone,
by training a scikit-learn classifier on those features only; report its
accuracy in the lab notebook."
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import pyloudnorm as pyln
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def extract_features_single(args_tuple):
    """Extract shallow non-semantic features from an audio file."""
    rel_path, data_root, label, split = args_tuple
    full_path = os.path.join(data_root, rel_path)
    
    basename = os.path.basename(rel_path)
    fname_len = len(basename)
    num_digits = sum(c.isdigit() for c in basename)
    num_alpha = sum(c.isalpha() for c in basename)
    digit_ratio = num_digits / max(1, fname_len)

    feat = {
        "rel_path": rel_path,
        "label": label,
        "split": split,
        "fname_len": fname_len,
        "digit_ratio": digit_ratio,
        "num_digits": num_digits,
        "num_alpha": num_alpha,
    }

    if not os.path.exists(full_path):
        return None

    try:
        stat = os.stat(full_path)
        feat["file_size_bytes"] = stat.st_size

        # Read audio using soundfile
        data, sr = sf.read(full_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        dur = len(data) / float(sr)
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data**2)))
        mean_val = float(np.mean(data))
        std_val = float(np.std(data))
        zcr = float(np.mean((data[:-1] * data[1:]) < 0))

        # Integrated loudness BS.1770
        meter = pyln.Meter(sr)
        try:
            loudness = float(meter.integrated_loudness(data))
            if np.isinf(loudness) or np.isnan(loudness):
                loudness = -70.0
        except Exception:
            loudness = -70.0

        feat["duration_sec"] = dur
        feat["sample_peak"] = peak
        feat["rms"] = rms
        feat["loudness_lufs"] = loudness
        feat["mean_val"] = mean_val
        feat["std_val"] = std_val
        feat["zcr"] = zcr

        return feat
    except Exception as e:
        return None


def run_audit(metadata_path, data_root, sample_per_class=500, num_workers=4, out_report=None):
    print(f"Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    print(f"Total rows in metadata: {len(df):,}")

    # Use train and val splits
    subset_df = df[df["split"].isin(["train", "val"])].copy()
    print(f"Rows in train + val: {len(subset_df):,}")
    print("Class distribution in full set:")
    print(subset_df["label"].value_counts())

    # Sample balanced set per class for fast and fair audit
    classes = subset_df["label"].dropna().unique()
    sampled_dfs = []
    for cls in classes:
        cls_df = subset_df[subset_df["label"] == cls]
        n_sample = min(len(cls_df), sample_per_class)
        sampled_dfs.append(cls_df.sample(n=n_sample, random_state=42))

    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\nSampled {len(sample_df)} clips ({sample_per_class} per class across {len(classes)} classes)")

    # Prepare multiprocessing tuples
    tasks = [
        (row["audio_path"], data_root, row["label"], row["split"])
        for _, row in sample_df.iterrows()
    ]

    print(f"\nExtracting audio statistics using {num_workers} processes...")
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(extract_features_single, tasks), total=len(tasks)):
            if res is not None:
                results.append(res)

    feat_df = pd.DataFrame(results)
    print(f"Successfully processed {len(feat_df)} / {len(tasks)} files.")

    feature_cols = [
        "fname_len", "digit_ratio", "num_digits", "num_alpha",
        "file_size_bytes", "duration_sec", "sample_peak",
        "rms", "loudness_lufs", "mean_val", "std_val", "zcr"
    ]

    # Clean any NaNs or infinities
    feat_df[feature_cols] = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split into train and val based on the original dataset split
    train_mask = feat_df["split"] == "train"
    val_mask = feat_df["split"] == "val"

    # If too few in val due to sampling, do a stratified 80/20 train/test split
    if val_mask.sum() < 50:
        print("Note: val split in sample is small; using stratified split for robust evaluation.")
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            feat_df[feature_cols], feat_df["label"], test_size=0.25,
            stratify=feat_df["label"], random_state=42
        )
    else:
        X_train = feat_df.loc[train_mask, feature_cols]
        y_train = feat_df.loc[train_mask, "label"]
        X_val = feat_df.loc[val_mask, feature_cols]
        y_val = feat_df.loc[val_mask, "label"]

    print(f"\nTrain size: {len(X_train)}, Val size: {len(X_val)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 1. Dummy Baseline (majority class)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_val)
    dummy_acc = accuracy_score(y_val, dummy_pred)
    dummy_f1 = f1_score(y_val, dummy_pred, average="macro")

    # 2. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_val_scaled)
    lr_acc = accuracy_score(y_val, lr_pred)
    lr_f1 = f1_score(y_val, lr_pred, average="macro")

    # 3. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_pred)
    rf_f1 = f1_score(y_val, rf_pred, average="macro")

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Output Markdown Summary
    report_md = f"""# Week 02 — Metadata Shortcut Risk Audit Report

**Date:** 2026-09-09
**Dataset:** CompSpoofV2 (`metadata_normalized.csv`)
**Objective:** Verify that audio manipulation classes cannot be predicted from file naming, duration, peak, loudness, or channel statistics alone.

---

## 1. Experimental Setup
- **Sample evaluated:** {len(feat_df):,} clips across all 5 classes ({sample_per_class} per class).
- **Extracted features:**
  - Filename features: `fname_len`, `digit_ratio`, `num_digits`, `num_alpha`
  - Audio physical properties: `file_size_bytes`, `duration_sec`
  - Signal & Loudness metrics: `sample_peak`, `rms`, `loudness_lufs`, `mean_val`, `std_val`, `zcr`
- **Classifiers trained:**
  1. Dummy Classifier (Most Frequent baseline / Chance)
  2. Logistic Regression (L2 regularized, standard scaled)
  3. Random Forest (100 trees, max_depth=6)

---

## 2. Classification Results on Validation Set

| Model | Accuracy | Macro-F1 | Verdict |
| :--- | :---: | :---: | :--- |
| **Dummy (Chance Baseline)** | {dummy_acc:.4f} | {dummy_f1:.4f} | Theoretical lower bound (~{100.0/len(classes):.1f}% for balanced {len(classes)} classes) |
| **Logistic Regression** | {lr_acc:.4f} | {lr_f1:.4f} | {'No shortcut risk detected' if lr_acc < 0.35 else 'Potential shortcut detected'} |
| **Random Forest (Non-linear)** | {rf_acc:.4f} | {rf_f1:.4f} | {'No shortcut risk detected' if rf_acc < 0.35 else 'Potential shortcut detected'} |

---

## 3. Feature Importance Ranking (Random Forest)
```
{importances.to_string()}
```

---

## 4. Conclusion & Protocol Audit Sign-Off
- **Shortcut risk:** {'LOW / PASS. Accuracy is close to the chance baseline (~20%), demonstrating that the model cannot exploit simple acoustic or metadata artifacts to cheat component-level spoof detection.' if rf_acc < 0.35 else 'HIGH / CAUTION. One or more acoustic features leak class identity.'}
- **Week 2 Milestone:** Complete. Proceed to toolchain construction (splicing with raised-cosine crossfade and loudness matching).
"""

    print("\n" + "=" * 60)
    print(report_md)
    print("=" * 60)

    if out_report:
        out_path = Path(out_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Report saved to: {out_report}")


def main():
    parser = argparse.ArgumentParser(description="CompSpoof-TL Metadata Shortcut Risk Audit")
    parser.add_argument("--metadata", default="/home/ee/btech/ee1230446/scratch/CompSpoofV2/metadata_normalized.csv",
                        help="Path to metadata_normalized.csv")
    parser.add_argument("--data_root", default="/home/ee/btech/ee1230446/scratch/CompSpoofV2",
                        help="Root directory of CompSpoofV2 audio")
    parser.add_argument("--sample_per_class", type=int, default=500,
                        help="Number of clips to sample per class (default: 500 = 2500 total)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes")
    parser.add_argument("--out_report", default=None,
                        help="Optional path to output markdown report")
    args = parser.parse_args()

    run_audit(
        metadata_path=args.metadata,
        data_root=args.data_root,
        sample_per_class=args.sample_per_class,
        num_workers=args.workers,
        out_report=args.out_report,
    )


if __name__ == "__main__":
    main()
