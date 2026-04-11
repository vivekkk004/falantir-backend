"""
Gemini Teacher — Auto-label surveillance videos for student-model training.

Falantir v2.1 training pipeline, Step 1 of 2:

    videos/  →  (this script)  →  labels.jsonl + frames/

Then Step 2 is `train_threat_classifier.py`, which loads the JSONL and
fine-tunes MobileNetV3 on Google Colab (free T4).

Usage:
    python training/label_with_gemini.py --videos_dir ./res \
                                         --output_dir ./dataset \
                                         --fps 0.5 \
                                         --max_frames_per_video 200

Cost estimate (Gemini 2.5 Flash, AI Studio free tier):
    - Free tier: 1,500 requests/day → ~1,500 labeled frames/day FREE
    - Paid: ~$0.0001 per frame → $1 = 10,000 frames, $10 = 100,000 frames

Output format (one JSON object per line, matches the student trainer's input):
    {
        "frame_path": "dataset/frames/vid01_0042.jpg",
        "video": "vid01.mp4",
        "frame_index": 42,
        "timestamp_s": 1.4,
        "threat_label": "suspicious",
        "threat_level": 1,
        "confidence": 0.82,
        "probabilities": {"safe": 0.1, "suspicious": 0.82, "critical": 0.08},
        "reasoning": "...",
        "scene_description": "...",
        "detected_objects": [...],
        "labeled_by": "gemini-2.5-flash",
        "labeled_at": "2026-04-11T12:34:56Z"
    }
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

# Allow running the script from the project root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.services.gemini_service import analyze_frame  # noqa: E402


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_videos(videos_dir: Path):
    for p in sorted(videos_dir.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS:
            yield p


def _safe_slug(text: str, max_len: int = 40) -> str:
    keep = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)
    return keep[:max_len].strip("_") or "video"


def label_video(
    video_path: Path,
    frames_dir: Path,
    target_fps: float,
    max_frames: int,
    jsonl_writer,
    stats: dict,
    jpeg_quality: int = 80,
    on_frame_labeled=None,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  SKIP: cannot open {video_path.name}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / max(target_fps, 0.01))))

    slug = _safe_slug(video_path.stem)
    video_frames_dir = frames_dir / slug
    video_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    labeled_here = 0

    while labeled_here < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        # Save raw frame to disk (for the student trainer to load later)
        frame_filename = f"{slug}_{frame_idx:06d}.jpg"
        frame_path = video_frames_dir / frame_filename
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        # Call Gemini teacher
        t0 = time.time()
        try:
            result = analyze_frame(frame)
        except Exception as e:
            print(f"  ! Gemini error on frame {frame_idx}: {e}")
            frame_idx += 1
            continue

        if result.get("error"):
            print(f"  ! Gemini fallback at frame {frame_idx}: {result['error']}")
            # Don't write fallback rows to the dataset — they'd poison training
            frame_idx += 1
            continue

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        record = {
            "frame_path": str(frame_path.relative_to(frames_dir.parent)).replace("\\", "/"),
            "video": video_path.name,
            "frame_index": frame_idx,
            "timestamp_s": round(frame_idx / src_fps, 3),
            "threat_label": result["threat_label"],
            "threat_level": result["threat_level"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "reasoning": result.get("reasoning", ""),
            "scene_description": result.get("scene_description", ""),
            "detected_objects": result.get("detected_objects", []),
            "labeled_by": result.get("model", "gemini-2.5-flash"),
            "labeled_at": datetime.now(timezone.utc).isoformat(),
            "teacher_latency_ms": elapsed_ms,
        }

        jsonl_writer.write(json.dumps(record) + "\n")
        jsonl_writer.flush()

        labeled_here += 1
        stats["total"] += 1
        stats["by_label"][result["threat_label"]] = stats["by_label"].get(result["threat_label"], 0) + 1

        if on_frame_labeled:
            on_frame_labeled(labeled_here, record)
        else:
            print(
                f"  [{labeled_here:>4}/{max_frames}] frame {frame_idx:>6} "
                f"→ {result['threat_label']:<11} ({result['confidence']:.0%}) "
                f"{elapsed_ms:>6}ms"
            )

        frame_idx += 1

    cap.release()
    return labeled_here


def main():
    parser = argparse.ArgumentParser(description="Auto-label videos with Gemini 2.5 Flash.")
    parser.add_argument("--videos_dir", type=str, required=True, help="Folder of input videos")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Where to write labels + frames")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second to sample")
    parser.add_argument("--max_frames_per_video", type=int, default=200, help="Hard cap per video")
    parser.add_argument("--daily_limit", type=int, default=1400, help="Stop after N total frames (free tier = 1500/day)")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    frames_dir = output_dir / "frames"
    labels_path = output_dir / "labels.jsonl"

    if not videos_dir.exists():
        print(f"ERROR: videos_dir {videos_dir} does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not os.getenv("GEMINI_API_KEY", "").strip():
        print("ERROR: GEMINI_API_KEY is not set — cannot label without the teacher model.")
        sys.exit(1)

    print("=" * 60)
    print("  FALANTIR — Gemini Teacher Auto-Labeler")
    print("=" * 60)
    print(f"  videos_dir : {videos_dir}")
    print(f"  output_dir : {output_dir}")
    print(f"  sample fps : {args.fps}")
    print(f"  per-video  : {args.max_frames_per_video}")
    print(f"  daily cap  : {args.daily_limit}")
    print("=" * 60)

    stats = {"total": 0, "by_label": {}}
    append_mode = labels_path.exists()
    mode = "a" if append_mode else "w"
    if append_mode:
        print(f"Appending to existing {labels_path}")

    with labels_path.open(mode, encoding="utf-8") as writer:
        for video_path in iter_videos(videos_dir):
            if stats["total"] >= args.daily_limit:
                print(f"\nReached daily limit of {args.daily_limit} frames. Stop.")
                break

            remaining = args.daily_limit - stats["total"]
            per_video_cap = min(args.max_frames_per_video, remaining)

            print(f"\n▶ {video_path.name}  (cap={per_video_cap})")
            label_video(
                video_path=video_path,
                frames_dir=frames_dir,
                target_fps=args.fps,
                max_frames=per_video_cap,
                jsonl_writer=writer,
                stats=stats,
            )

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
    print(f"  Total labeled : {stats['total']}")
    print(f"  By label      : {stats['by_label']}")
    print(f"  Dataset file  : {labels_path}")
    print(f"  Frame folder  : {frames_dir}")
    print("\nNext step:")
    print("  python training/train_threat_classifier.py \\")
    print(f"      --dataset {labels_path} \\")
    print("      --epochs 15 --batch_size 32")


if __name__ == "__main__":
    main()
