"""
Falantir v2.1 — DCSASS → ImageFolder preprocessing (uses CSV labels).

Converts the DCSASS Kaggle dataset into the ImageFolder layout that
train_threat_classifier.py expects, using the official Labels/*.csv
files to split each class into normal (0) and anomalous (1) subsets.

Input structure expected:
    dataset/
      Shoplifting/
        Shoplifting001_x264.mp4/
          Shoplifting001_x264_0.mp4
          Shoplifting001_x264_1.mp4
          ...
      Labels/
        Shoplifting.csv     (rows: clip_id, class, label)
        Robbery.csv         (optional — more classes)
        ...

Label mapping (CSV label 0/1 → our 3-class system):
    Shoplifting  0 → safe,  1 → suspicious
    Stealing     0 → safe,  1 → suspicious
    Burglary     0 → safe,  1 → suspicious
    Vandalism    0 → safe,  1 → suspicious
    Robbery      0 → safe,  1 → critical
    Assault      0 → safe,  1 → critical
    Fighting     0 → safe,  1 → critical
    Shooting     0 → safe,  1 → critical

Output:
    dataset_imagefolder/
      train/
        safe/, suspicious/, critical/
      val/
        safe/, suspicious/, critical/

Usage:
    python training/prepare_dcsass.py
    python training/prepare_dcsass.py --frames_safe 2 --frames_anomaly 8
    python training/prepare_dcsass.py --val_split 0.15
"""

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np


# For each DCSASS class, define how CSV label 0/1 maps to our 3-class taxonomy
CSV_CLASS_MAP = {
    "Shoplifting": {"0": "safe", "1": "suspicious"},
    "Stealing":    {"0": "safe", "1": "suspicious"},
    "Burglary":    {"0": "safe", "1": "suspicious"},
    "Vandalism":   {"0": "safe", "1": "suspicious"},
    "Robbery":     {"0": "safe", "1": "critical"},
    "Assault":     {"0": "safe", "1": "critical"},
    "Fighting":    {"0": "safe", "1": "critical"},
    "Shooting":    {"0": "safe", "1": "critical"},
    "Abuse":       {"0": "safe", "1": "critical"},
    "Arson":       {"0": "safe", "1": "critical"},
    "Explosion":   {"0": "safe", "1": "critical"},
}


def _stable_split(key: str, val_fraction: float) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return "val" if (h % 1000) / 1000.0 < val_fraction else "train"


def _is_black_frame(frame, threshold: float = 0.02) -> bool:
    if frame is None:
        return True
    return float(np.mean(frame)) < threshold * 255


def _extract_frames(clip_path: Path, n_frames: int):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    margin = min(3, total // 10)
    usable = max(1, total - 2 * margin)
    if n_frames >= usable:
        indices = list(range(margin, margin + usable))
    else:
        step = usable / n_frames
        indices = [margin + int(i * step) for i in range(n_frames)]

    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or _is_black_frame(frame):
            continue
        out.append((idx, frame))
    cap.release()
    return out


def _find_clip_path(dcsass_root: Path, class_folder: str, clip_id: str) -> Path | None:
    """
    Given clip_id like 'Shoplifting001_x264_9', figure out the nested path:
      dcsass_root/Shoplifting/Shoplifting001_x264.mp4/Shoplifting001_x264_9.mp4
    """
    parts = clip_id.rsplit("_", 1)
    if len(parts) != 2:
        return None
    parent_name = f"{parts[0]}.mp4"  # e.g. Shoplifting001_x264.mp4 (this is a FOLDER)
    clip_name = f"{clip_id}.mp4"     # e.g. Shoplifting001_x264_9.mp4
    candidate = dcsass_root / class_folder / parent_name / clip_name
    return candidate if candidate.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcsass_root", default="dataset")
    ap.add_argument("--labels_dir", default="dataset/Labels")
    ap.add_argument("--out", default="dataset_imagefolder")
    ap.add_argument("--frames_safe", type=int, default=2,
                    help="Frames per CSV-label-0 (normal) clip")
    ap.add_argument("--frames_anomaly", type=int, default=8,
                    help="Frames per CSV-label-1 (anomaly) clip — more because rarer")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--jpeg_quality", type=int, default=88)
    ap.add_argument("--max_per_class", type=int, default=5000,
                    help="Cap total frames per output class (0 = no cap)")
    args = ap.parse_args()

    dcsass_root = Path(args.dcsass_root).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    out_root = Path(args.out).resolve()

    # Which CSVs exist + which classes are actually downloaded as video?
    available_csvs = {}
    for csv_path in labels_dir.glob("*.csv"):
        class_name = csv_path.stem
        if (dcsass_root / class_name).is_dir():
            available_csvs[class_name] = csv_path

    if not available_csvs:
        print("ERROR: No labeled class folders found. You need matching video folder AND Labels CSV.")
        print(f"  Looked under : {dcsass_root}")
        print(f"  Labels dir   : {labels_dir}")
        return

    print("=" * 70)
    print("  DCSASS → ImageFolder preprocessing")
    print("=" * 70)
    print(f"  DCSASS root    : {dcsass_root}")
    print(f"  Labels dir     : {labels_dir}")
    print(f"  Output root    : {out_root}")
    print(f"  Frames/safe    : {args.frames_safe}")
    print(f"  Frames/anomaly : {args.frames_anomaly}")
    print(f"  Val split      : {args.val_split}")
    print(f"  Cap/class      : {args.max_per_class or 'none'}")
    print()
    print(f"  Classes with both videos AND labels: {list(available_csvs.keys())}")
    print("=" * 70)
    print()

    target_classes = sorted({
        mapped
        for cls in available_csvs
        for mapped in CSV_CLASS_MAP.get(cls, {}).values()
    })
    for split in ("train", "val"):
        for cls in target_classes:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {split: {cls: 0 for cls in target_classes} for split in ("train", "val")}

    def _at_cap(split, cls):
        if not args.max_per_class:
            return False
        allowed = int(args.max_per_class * (1 - args.val_split if split == "train" else args.val_split))
        return counts[split][cls] >= allowed

    def _save(frame, mapped_class, clip_id, frame_idx):
        split = _stable_split(f"{clip_id}#{frame_idx}", args.val_split)
        if _at_cap(split, mapped_class):
            return False
        fname = f"{clip_id}_f{frame_idx:05d}.jpg"
        dst = out_root / split / mapped_class / fname
        if dst.exists():
            counts[split][mapped_class] += 1
            return True
        cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        counts[split][mapped_class] += 1
        return True

    # ─── Process each CSV ──────────────────────────────────
    for class_name, csv_path in available_csvs.items():
        mapping = CSV_CLASS_MAP.get(class_name)
        if not mapping:
            print(f"  SKIP {class_name}: no class mapping defined")
            continue

        print(f"▶ Processing '{class_name}' → label0={mapping['0']}, label1={mapping['1']}")

        clip_rows = []
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                clip_id, _, label = row[0].strip(), row[1].strip(), row[2].strip()
                if label not in ("0", "1"):
                    continue
                clip_rows.append((clip_id, label))

        n_found = 0
        n_missing = 0
        frames_total = 0

        for i, (clip_id, label) in enumerate(clip_rows, 1):
            path = _find_clip_path(dcsass_root, class_name, clip_id)
            if path is None:
                n_missing += 1
                continue
            mapped = mapping[label]
            n_frames = args.frames_anomaly if label == "1" else args.frames_safe
            frames = _extract_frames(path, n_frames)
            for idx, frame in frames:
                if _save(frame, mapped, clip_id, idx):
                    frames_total += 1
            n_found += 1

            if i % 100 == 0:
                print(f"    {i}/{len(clip_rows)} clips processed ({frames_total} frames so far)")

        print(f"    Found {n_found} clips, missing {n_missing}, extracted {frames_total} frames")
        print()

    # ─── Report ──────────────────────────────────
    print("=" * 70)
    print("  Preprocessing complete")
    print("=" * 70)
    for split in ("train", "val"):
        total = sum(counts[split].values())
        print(f"  {split:5} total : {total}")
        for cls in target_classes:
            n = counts[split][cls]
            pct = (100.0 * n / total) if total else 0.0
            bar = "█" * int(pct / 2.5)
            print(f"    {cls:<12}: {n:>6}  ({pct:5.1f}%)  {bar}")
        print()
    grand = sum(sum(counts[s].values()) for s in ("train", "val"))
    print(f"  Grand total  : {grand} frames across {len(target_classes)} classes")
    print(f"  Output       : {out_root}")
    print()
    print("Next: zip dataset_imagefolder/, upload to Google Drive, then train on Colab.")


if __name__ == "__main__":
    main()
