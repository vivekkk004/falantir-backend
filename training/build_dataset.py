"""
Falantir v2.1 — Dataset builder (JSONL → ImageFolder)

Bridges label_with_gemini.py → train_threat_classifier.py.

Reads the JSONL file produced by label_with_gemini.py, splits frames
into train/val, and copies each frame into the ImageFolder layout that
train_threat_classifier.py expects:

    dataset/
      train/
        safe/          *.jpg
        suspicious/    *.jpg
        critical/      *.jpg
      val/
        safe/          *.jpg
        suspicious/    *.jpg
        critical/      *.jpg

Usage:
    python training/build_dataset.py \
        --labels dataset/labels.jsonl \
        --out dataset_imagefolder \
        --val_split 0.15 \
        --min_confidence 0.5

Notes:
    - Uses the frame_path field from each JSONL record.
    - Drops any record whose confidence is below --min_confidence
      (Gemini's fallback 'safe' labels come with 0.0 confidence and
      would poison training).
    - Deterministic split by hashing the frame_path so re-runs are stable.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path


LABELS = ("safe", "suspicious", "critical")


def _stable_split(key: str, val_fraction: float) -> str:
    """Return 'val' or 'train' based on a stable hash of the key."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return "val" if (h % 1000) / 1000.0 < val_fraction else "train"


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL labels to ImageFolder dataset.")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels.jsonl")
    parser.add_argument("--out", type=str, default="dataset_imagefolder", help="Output root directory")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction of frames for validation")
    parser.add_argument("--min_confidence", type=float, default=0.5,
                        help="Drop records with confidence below this threshold")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base dir frame_path is relative to (default: parent of --labels)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files (default: hardlink when possible, falls back to copy)")
    args = parser.parse_args()

    labels_path = Path(args.labels).resolve()
    if not labels_path.exists():
        print(f"ERROR: labels file not found — {labels_path}")
        sys.exit(1)

    base_dir = Path(args.base_dir).resolve() if args.base_dir else labels_path.parent
    out_root = Path(args.out).resolve()

    # Prep ImageFolder structure
    for split in ("train", "val"):
        for label in LABELS:
            (out_root / split / label).mkdir(parents=True, exist_ok=True)

    counts = {split: Counter() for split in ("train", "val")}
    dropped_low_conf = 0
    dropped_missing = 0
    total_records = 0

    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_records += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            label = rec.get("threat_label")
            if label not in LABELS:
                continue

            confidence = float(rec.get("confidence", 0.0))
            if confidence < args.min_confidence:
                dropped_low_conf += 1
                continue

            frame_rel = rec.get("frame_path")
            if not frame_rel:
                continue

            # Resolve frame path. frame_path in JSONL is relative to dataset/
            # (the parent of labels.jsonl), per label_with_gemini.py.
            src = (base_dir / frame_rel).resolve()
            if not src.exists():
                dropped_missing += 1
                continue

            split = _stable_split(frame_rel, args.val_split)
            dst_name = src.name
            dst = out_root / split / label / dst_name

            # Skip duplicates silently
            if dst.exists():
                counts[split][label] += 1
                continue

            try:
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        os.link(src, dst)  # hardlink — free, instant
                    except OSError:
                        shutil.copy2(src, dst)
            except OSError as e:
                print(f"  ! failed to place {src.name}: {e}")
                continue

            counts[split][label] += 1

    # ─── Report ───
    print("=" * 60)
    print("  Dataset build complete")
    print("=" * 60)
    print(f"  labels.jsonl records : {total_records}")
    print(f"  dropped (low conf)   : {dropped_low_conf}")
    print(f"  dropped (missing)    : {dropped_missing}")
    print()
    for split in ("train", "val"):
        total = sum(counts[split].values())
        print(f"  {split:5} total       : {total}")
        for label in LABELS:
            n = counts[split][label]
            pct = (100.0 * n / total) if total else 0.0
            print(f"    {label:<11}: {n:>5}  ({pct:5.1f}%)")
    print()
    print(f"  Output directory     : {out_root}")
    print()
    print("Next step:")
    print(f"  python training/train_threat_classifier.py  "
          f"# edit DATASET_DIR to: {out_root}")


if __name__ == "__main__":
    main()
