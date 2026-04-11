"""
Inference Pipeline — Falantir v2.1

In v2, this module ran three models in parallel (YOLO + Gemini + MobileNetV3).
In v2.1 we delegate to the VisionProvider abstraction, which handles the
fallback chain (Gemini → MobileNetV3 → SafeFallback) transparently.

The public API (`analyze_frame`, `get_models_status`, `load_all_models`)
stays the same so stream_service.py and detection_routes_v2.py don't break.
"""

import os
import time
import cv2
from dotenv import load_dotenv

from api.services.vision_provider import (
    analyze_frame as provider_analyze,
    get_providers_status,
    warmup,
)

load_dotenv()

# Kept for backward compat with routes/services that still read these.
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))
THREAT_THRESHOLD = float(os.getenv("THREAT_THRESHOLD", "0.6"))

_models_loaded = False


def load_all_models():
    """Eagerly initialize the active vision provider at app startup."""
    global _models_loaded
    if _models_loaded:
        return

    print("=" * 50)
    print("  Loading vision pipeline (v2.1)")
    print("=" * 50)
    warmup()
    _models_loaded = True
    print("=" * 50)
    print("  Vision pipeline ready")
    print("=" * 50)


def get_models_status():
    """Status payload for GET /api/detection/models."""
    if not _models_loaded:
        try:
            load_all_models()
        except Exception as e:
            print(f"WARMUP ERROR: {e}")

    providers = get_providers_status()

    # Individual provider details
    from api.services.gemini_service import get_status as gemini_status

    # MobileNetV3 detail
    try:
        from api.models.threat_classifier import _model as mnet_model, _device
        mnet_detail = {
            "loaded": mnet_model is not None,
            "device": str(_device) if _device else "N/A",
            "model_file": os.path.basename(os.getenv("MODEL_PATH", "threat_classifier.pt")),
            "role": "offline fallback (trained student model)",
        }
    except Exception:
        mnet_detail = {"loaded": False, "role": "offline fallback (trained student model)"}

    return {
        "active_provider": providers["active_provider"],
        "fallback_chain": providers["chain"],
        "providers": providers["providers"],
        "gemini": gemini_status(),
        "mobilenetv3": mnet_detail,
        "motion_detection": {
            "enabled": True,
            "threshold": float(os.getenv("MOTION_THRESHOLD", "0.015")),
            "cooldown_frames": int(os.getenv("MOTION_COOLDOWN_FRAMES", "15")),
        },
    }


# ─── bbox helpers ─────────────────────────────────────────

def _draw_objects(frame_bgr, detected_objects):
    """Draw bounding boxes on a frame. bbox is [ymin, xmin, ymax, xmax] 0-1000."""
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    for obj in detected_objects:
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4:
            continue

        try:
            ymin, xmin, ymax, xmax = bbox
        except (TypeError, ValueError):
            continue

        # Normalized 0-1000 → pixel coords
        x1 = int(xmin / 1000.0 * w)
        y1 = int(ymin / 1000.0 * h)
        x2 = int(xmax / 1000.0 * w)
        y2 = int(ymax / 1000.0 * h)

        if x2 <= x1 or y2 <= y1:
            continue

        label = obj.get("label", "object")
        conf = obj.get("confidence", 0.0)
        action = obj.get("action", "")

        # Color by whether the object has a notable action
        color = (0, 150, 255) if action else (0, 200, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%}"
        if action:
            text = f"{label}: {action[:30]}"

        # Text background for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


def _draw_threat_overlay(frame_bgr, threat_label, confidence):
    """Top-left HUD showing the overall threat level."""
    label = (threat_label or "safe").upper()
    color_map = {
        "SAFE": (0, 200, 0),
        "SUSPICIOUS": (0, 165, 255),
        "CRITICAL": (0, 0, 255),
    }
    color = color_map.get(label, (200, 200, 200))

    text = f"{label} ({confidence:.0%})"
    cv2.putText(
        frame_bgr, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
    )
    return frame_bgr


# ─── Main Entry Point ─────────────────────────────────────

def analyze_frame(frame_bgr, **_legacy_kwargs):
    """
    Run the active vision provider on a single frame and annotate it.

    Args:
        frame_bgr: OpenCV BGR image.
        **_legacy_kwargs: absorbed for backward compat with callers that still
            pass `run_gemini=...`. Ignored in v2.1 — there is only one provider
            call per frame, gated upstream by motion detection.

    Returns:
        dict with:
            detected_objects, scene_description, reasoning,
            threat_label, threat_level, confidence, probabilities,
            annotated_frame, inference_time_ms, provider_used, model

        Plus legacy alias: yolo_objects (same as detected_objects) for any
        un-migrated frontend code.
    """
    start = time.time()
    result = provider_analyze(frame_bgr)

    detected_objects = result.get("detected_objects", [])
    threat_label = result.get("threat_label", "safe")
    confidence = result.get("confidence", 0.0)

    # Annotate frame
    annotated = _draw_objects(frame_bgr, detected_objects)
    annotated = _draw_threat_overlay(annotated, threat_label, confidence)

    total_ms = round((time.time() - start) * 1000, 1)

    return {
        "detected_objects": detected_objects,
        "yolo_objects": detected_objects,  # legacy alias
        "scene_description": result.get("scene_description", ""),
        "gemini_description": result.get("scene_description", ""),  # legacy alias
        "reasoning": result.get("reasoning", ""),
        "threat_label": threat_label,
        "threat_level": result.get("threat_level", 0),
        "confidence": confidence,
        "probabilities": result.get("probabilities", {
            "safe": 1.0, "suspicious": 0.0, "critical": 0.0,
        }),
        "provider_used": result.get("provider_used", "unknown"),
        "model": result.get("model", "unknown"),
        "annotated_frame": annotated,
        "inference_time_ms": result.get("inference_time_ms", total_ms),
    }
