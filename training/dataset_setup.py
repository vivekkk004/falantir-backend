"""
Falantir v2 — Dataset setup helper.

Creates the expected folder structure and provides utilities for:
- Extracting frames from UCF-Crime videos
- Organizing frames into safe/suspicious/critical folders
- Data augmentation helpers
"""

import os
import cv2
import random
import shutil


DATASET_DIR = "dataset"
CATEGORIES = ["safe", "suspicious", "critical"]


def create_dataset_structure():
    """Create the train/val folder structure."""
    for split in ["train", "val"]:
        for cat in CATEGORIES:
            path = os.path.join(DATASET_DIR, split, cat)
            os.makedirs(path, exist_ok=True)
            print(f"Created: {path}")
    print("Dataset structure ready!")


def extract_frames_from_video(video_path, output_dir, every_n=10, max_frames=200):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        every_n: Extract every Nth frame
        max_frames: Maximum number of frames to extract
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return 0

    frame_count = 0
    saved = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % every_n != 0:
            continue

        # Resize to 640px width
        h, w = frame.shape[:2]
        ratio = 640 / float(w)
        frame = cv2.resize(frame, (640, int(h * ratio)))

        filename = f"{video_name}_frame_{frame_count:06d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        saved += 1

    cap.release()
    print(f"Extracted {saved} frames from {video_path}")
    return saved


def split_train_val(source_dir, category, train_ratio=0.8):
    """
    Split frames from source_dir into train/val for a given category.

    Args:
        source_dir: Directory containing extracted frames
        category: One of "safe", "suspicious", "critical"
        train_ratio: Fraction for training (rest goes to validation)
    """
    files = [f for f in os.listdir(source_dir) if f.endswith((".jpg", ".png"))]
    random.shuffle(files)

    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    train_dir = os.path.join(DATASET_DIR, "train", category)
    val_dir = os.path.join(DATASET_DIR, "val", category)

    for f in train_files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(val_dir, f))

    print(f"{category}: {len(train_files)} train, {len(val_files)} val")


def count_dataset():
    """Print dataset statistics."""
    print("\nDataset Statistics:")
    print("-" * 40)
    for split in ["train", "val"]:
        for cat in CATEGORIES:
            path = os.path.join(DATASET_DIR, split, cat)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith((".jpg", ".png"))])
                print(f"  {split}/{cat}: {count} images")
    print("-" * 40)


if __name__ == "__main__":
    create_dataset_structure()
    print("\nNext steps:")
    print("1. Download UCF-Crime dataset and extract frames into categorized folders")
    print("2. Record custom footage and extract frames")
    print("3. Use extract_frames_from_video() to extract frames from videos")
    print("4. Use split_train_val() to split into train/val")
    print("5. Run train_threat_classifier.py to train the model")
