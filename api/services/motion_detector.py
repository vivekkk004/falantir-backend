"""
Motion Detector — Falantir v2.1

Cheap, local, free motion detection using OpenCV background subtraction.
This is Tier 0 of the cost-optimized inference pipeline: if nothing is
moving in the frame, we skip the expensive vision provider call entirely.

Typical behavior in a retail store:
  - Empty shop → ~99% of frames skipped
  - Busy shop → ~30-50% of frames skipped
  - Cost drops by an order of magnitude

The detector is per-agent (each camera gets its own instance) because
background models must not cross-contaminate between locations.
"""

import os
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class MotionDetector:
    """
    Background-subtraction motion detector.

    Uses MOG2 (Mixture of Gaussians) which adapts to slow lighting changes
    but triggers on real movement. Returns a boolean per frame plus the
    fraction of the frame that changed.
    """

    def __init__(
        self,
        threshold: float = None,
        history: int = 500,
        var_threshold: float = 25.0,
        min_area: int = 500,
        cooldown_frames: int = None,
    ):
        # Fraction of frame pixels (0-1) that must change to count as motion
        self.threshold = float(
            threshold if threshold is not None
            else os.getenv("MOTION_THRESHOLD", "0.015")
        )

        # Minimum contiguous blob area (px) to ignore camera noise
        self.min_area = int(min_area)

        # After motion triggers analysis, wait N frames before analyzing again
        # (rate limit so we don't burn API credit on continuous movement)
        self.cooldown_frames = int(
            cooldown_frames if cooldown_frames is not None
            else os.getenv("MOTION_COOLDOWN_FRAMES", "15")
        )

        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self._frames_since_trigger = self.cooldown_frames  # allow immediate first trigger
        self._warmup_frames_remaining = 10  # Ignore first 10 frames (bg adaptation)
        self._last_motion_ratio = 0.0

    def check(self, frame_bgr):
        """
        Process a frame and decide whether to run vision analysis on it.

        Returns (should_analyze: bool, motion_ratio: float).
        """
        self._frames_since_trigger += 1

        # Warmup phase — let background model learn
        if self._warmup_frames_remaining > 0:
            self._bg.apply(frame_bgr)
            self._warmup_frames_remaining -= 1
            return False, 0.0

        # Compute foreground mask
        mask = self._bg.apply(frame_bgr)

        # Clean mask — morphological open removes noise specks
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Count motion pixels (foreground = 255)
        motion_pixels = int((mask > 127).sum())
        total_pixels = mask.size
        motion_ratio = motion_pixels / float(total_pixels)
        self._last_motion_ratio = motion_ratio

        has_motion = motion_ratio > self.threshold

        # Cooldown check — even if motion, don't re-trigger too fast
        if not has_motion:
            return False, motion_ratio

        if self._frames_since_trigger < self.cooldown_frames:
            return False, motion_ratio

        # Triggered — reset cooldown
        self._frames_since_trigger = 0
        return True, motion_ratio

    @property
    def last_motion_ratio(self) -> float:
        return self._last_motion_ratio
