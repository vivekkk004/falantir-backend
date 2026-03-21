"""
Three-Model Parallel Inference Pipeline.

Runs YOLO, Gemini, and the custom threat classifier simultaneously
on every frame using Python threading. The combined result is returned
as fast as the slowest model.
"""

import os
import threading
import time
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─── YOLO Setup ───────────────────────────────────────────

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

_yolo_model = None
_models_loaded = False
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "configs", "shoplifting_wights.pt")

FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))
THREAT_THRESHOLD = float(os.getenv("THREAT_THRESHOLD", "0.6"))


def _ensure_models_loaded():
    """Lazy-load models if not already loaded (handles Flask debug reloader)."""
    global _yolo_model, _models_loaded
    if _models_loaded:
        return

    if _HAS_YOLO and os.path.exists(YOLO_MODEL_PATH) and _yolo_model is None:
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO: Lazy-loaded from {YOLO_MODEL_PATH}")

    from api.models.threat_classifier import load_model
    load_model()

    from api.services.gemini_service import _get_client
    _get_client()

    _models_loaded = True


def load_all_models():
    """Load all three models once at startup."""
    global _yolo_model, _models_loaded

    # 1. Load YOLO
    if _HAS_YOLO and os.path.exists(YOLO_MODEL_PATH):
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO: Loaded from {YOLO_MODEL_PATH}")
    else:
        print("YOLO: Not available — object detection disabled")

    # 2. Load custom threat classifier
    from api.models.threat_classifier import load_model
    model, device = load_model()
    if model:
        print(f"THREAT MODEL: Ready on {device}")

    # 3. Initialize Gemini client
    from api.services.gemini_service import _get_client
    client = _get_client()
    if client:
        print("GEMINI: Ready")

    _models_loaded = True
    print("=== All models loaded ===")


def get_models_status():
    """Return status of all three models."""
    _ensure_models_loaded()
    from api.models.threat_classifier import _model as threat_model, _device
    from api.services.gemini_service import get_status as gemini_status

    return {
        "yolo": {
            "loaded": _yolo_model is not None,
            "version": "YOLOv8",
            "model_file": os.path.basename(YOLO_MODEL_PATH),
            "device": "cpu",
        },
        "gemini": gemini_status(),
        "threat_classifier": {
            "loaded": threat_model is not None,
            "model_file": os.path.basename(os.getenv("MODEL_PATH", "threat_classifier.pt")),
            "device": str(_device) if _device else "N/A",
        },
    }


# ─── YOLO Inference ───────────────────────────────────────

def _run_yolo(frame, result_dict):
    """Run YOLO on frame, store objects list in result_dict["yolo"]."""
    _ensure_models_loaded()

    if _yolo_model is None:
        result_dict["yolo"] = {"objects": [], "annotated_frame": frame}
        return

    try:
        results = _yolo_model.predict(frame, verbose=False)
        boxes = results[0].boxes if (results and results[0].boxes) else None

        objects = []
        annotated = frame.copy()

        if boxes is not None:
            data = np.array(boxes.data)
            xyxy = np.array(boxes.xyxy).astype("int32")

            for (x1, y1, x2, y2), (*_, conf, cls) in zip(xyxy, data):
                cls_id = int(cls)
                conf_val = float(conf)
                label = "Shoplifting" if cls_id == 1 else "Normal"

                objects.append({
                    "label": label,
                    "class_id": cls_id,
                    "confidence": round(conf_val, 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

                # Draw bounding boxes
                if cls_id == 1:
                    color = (0, 0, 255)  # Red for shoplifting
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"ALERT {conf_val:.0%}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif conf_val > 0.5:
                    color = (0, 255, 0)  # Green for normal
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

        result_dict["yolo"] = {
            "objects": objects,
            "annotated_frame": annotated,
        }
    except Exception as e:
        print(f"YOLO ERROR: {e}")
        result_dict["yolo"] = {"objects": [], "annotated_frame": frame}


def _run_gemini(frame, result_dict):
    """Run Gemini scene description, store in result_dict["gemini"]."""
    try:
        from api.services.gemini_service import describe_frame
        result = describe_frame(frame)
        result_dict["gemini"] = result or {"description": "Scene analysis unavailable", "model": "gemini-1.5-flash"}
    except Exception as e:
        print(f"GEMINI ERROR: {e}")
        result_dict["gemini"] = {"description": f"Error: {str(e)[:60]}", "model": "gemini-1.5-flash"}


def _run_threat_classifier(frame, result_dict):
    """Run custom threat classifier, store in result_dict["threat"]."""
    try:
        from api.models.threat_classifier import classify_frame
        result = classify_frame(frame)
        result_dict["threat"] = result or {
            "threat_label": "safe",
            "threat_level": 0,
            "confidence": 0.0,
            "probabilities": {"safe": 1.0, "suspicious": 0.0, "critical": 0.0},
        }
    except Exception as e:
        print(f"THREAT CLASSIFIER ERROR: {e}")
        result_dict["threat"] = {
            "threat_label": "safe",
            "threat_level": 0,
            "confidence": 0.0,
            "probabilities": {"safe": 1.0, "suspicious": 0.0, "critical": 0.0},
        }


# ─── Combined Parallel Inference ──────────────────────────

def analyze_frame(frame_bgr, run_gemini=True):
    """
    Run all three models in parallel on a single frame.

    Returns dict with yolo_objects, gemini_description, threat info, annotated_frame, timing.
    """
    _ensure_models_loaded()
    start_time = time.time()
    result = {}

    # Launch all models in parallel
    threads = [
        threading.Thread(target=_run_yolo, args=(frame_bgr, result)),
        threading.Thread(target=_run_threat_classifier, args=(frame_bgr, result)),
    ]

    if run_gemini:
        threads.append(threading.Thread(target=_run_gemini, args=(frame_bgr, result)))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    # Combine results
    yolo = result.get("yolo", {"objects": [], "annotated_frame": frame_bgr})
    gemini = result.get("gemini", {"description": "Skipped (frame skip)", "model": "gemini-1.5-flash"})
    threat = result.get("threat", {
        "threat_label": "safe", "threat_level": 0,
        "confidence": 0.0, "probabilities": {"safe": 1.0, "suspicious": 0.0, "critical": 0.0},
    })

    # Use YOLO detections to boost threat level if shoplifting detected
    shoplifting_objects = [o for o in yolo["objects"] if o["class_id"] == 1]
    if shoplifting_objects and threat["threat_level"] == 0:
        best_conf = max(o["confidence"] for o in shoplifting_objects)
        threat["threat_label"] = "critical" if best_conf > 0.7 else "suspicious"
        threat["threat_level"] = 2 if best_conf > 0.7 else 1
        threat["confidence"] = best_conf
        threat["probabilities"] = {
            "safe": round(1 - best_conf, 4),
            "suspicious": round(best_conf if best_conf <= 0.7 else 0, 4),
            "critical": round(best_conf if best_conf > 0.7 else 0, 4),
        }

    # Draw threat overlay on annotated frame
    annotated = yolo["annotated_frame"]
    label = threat["threat_label"].upper()
    conf = threat["confidence"]
    color_map = {"SAFE": (0, 200, 0), "SUSPICIOUS": (0, 200, 255), "CRITICAL": (0, 0, 255)}
    color = color_map.get(label, (200, 200, 200))

    cv2.putText(annotated, f"{label} ({conf:.0%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    elapsed = (time.time() - start_time) * 1000

    return {
        "yolo_objects": yolo["objects"],
        "gemini_description": gemini["description"],
        "gemini_model": gemini.get("model", "gemini-1.5-flash"),
        "threat_label": threat["threat_label"],
        "threat_level": threat["threat_level"],
        "confidence": threat["confidence"],
        "probabilities": threat["probabilities"],
        "annotated_frame": annotated,
        "inference_time_ms": round(elapsed, 1),
    }
