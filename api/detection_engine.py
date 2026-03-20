import threading
import time
import base64
import os
import cv2
import numpy as np
from datetime import datetime, timezone
from collections import deque
from api.database import alerts_collection
import asyncio

# Attempt imports for YOLO
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

# Configuration and Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "configs", "shoplifting_wights.pt")
WIDTH = 640

# Colors (BGR)
CLS0_COLOR = (0, 255, 255)
CLS1_COLOR = (0, 255, 0)
STATUS_COLOR = (0, 0, 255)

# ─── Singleton State ──────────────────────────────────────

_lock = threading.Lock()
_thread: threading.Thread | None = None
_running = False
_latest_frame: bytes | None = None
_current_source = None
latest_alerts: deque = deque(maxlen=20)

_stats = {
    "total_frames": 0,
    "shoplifting_count": 0,
    "not_shoplifting_count": 0,
    "started_at": datetime.now(timezone.utc).isoformat(),
}

# ─── Internal Helpers ─────────────────────────────────────

def _encode_jpeg(frame) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()

def _make_placeholder(msg="Connecting...") -> bytes:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, msg, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    return _encode_jpeg(img)

def _save_alert_to_db(alert_data):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Ensure deep copy and serialization
        db_data = alert_data.copy()
        if isinstance(db_data.get("timestamp"), str):
            db_data["timestamp"] = datetime.fromisoformat(db_data["timestamp"])
        
        loop.run_until_complete(alerts_collection.insert_one(db_data))
        loop.close()
    except Exception as e:
        print(f"ALERTS-DB ERROR: {e}")

# ─── Detection Loop ───────────────────────────────────────

def _detection_loop(source):
    global _running, _latest_frame, _stats

    if not _HAS_YOLO:
        _latest_frame = _make_placeholder("ultralytics not installed")
        _running = False
        return

    if not os.path.exists(MODEL_PATH):
        _latest_frame = _make_placeholder("Model weights missing")
        _running = False
        return

    model = YOLO(MODEL_PATH)
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        _latest_frame = _make_placeholder("Source not found")
        _running = False
        return

    # Reset stats for new session
    _stats["total_frames"] = 0
    _stats["shoplifting_count"] = 0
    _stats["not_shoplifting_count"] = 0
    _stats["started_at"] = datetime.now(timezone.utc).isoformat()

    while _running:
        ret, frame = cap.read()
        if not ret:
            if not str(source).isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        h, w = frame.shape[:2]
        ratio = WIDTH / float(w)
        frame = cv2.resize(frame, (WIDTH, int(h * ratio)))
        _stats["total_frames"] += 1

        results = model.predict(frame, verbose=False)
        boxes = results[0].boxes if (results and results[0].boxes) else None
        
        status_text = "Scanning..."
        
        if boxes:
            data = np.array(boxes.data)
            xyxy = np.array(boxes.xyxy).astype("int32")
            
            for (x1, y1, x2, y2), (*_, conf, cls) in zip(xyxy, data):
                cls = int(cls)
                conf_val = float(conf)
                
                if cls == 1:
                    status_text = "Shoplifting!"
                    _stats["shoplifting_count"] += 1
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), CLS1_COLOR, 2)
                    center_x = (x1 + x2) // 2
                    cv2.circle(frame, (center_x, y1), 6, (0, 0, 255), -1)
                    
                    # Periodic alert saving
                    if _stats["total_frames"] % 40 == 0:
                        _, snap_buf = cv2.imencode(".jpg", frame)
                        alert = {
                            "camera_id": str(source),
                            "camera_label": f"Camera {source}",
                            "status": "Shoplifting",
                            "confidence": round(conf_val * 100, 1),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "snapshot": base64.b64encode(snap_buf).decode()
                        }
                        latest_alerts.appendleft(alert)
                        threading.Thread(target=_save_alert_to_db, args=(alert,), daemon=True).start()

                elif cls == 0 and conf_val > 0.8:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), CLS0_COLOR, 1)
                    _stats["not_shoplifting_count"] += 1

        cv2.putText(frame, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, STATUS_COLOR, 2)

        with _lock:
            _latest_frame = _encode_jpeg(frame)
        
        time.sleep(0.01)

    cap.release()
    _running = False

# ─── Public API ───────────────────────────────────────────

def start(source="0"):
    global _thread, _running, _current_source
    if _running: return False
    _running = True
    _current_source = source
    _thread = threading.Thread(target=_detection_loop, args=(source,), daemon=True)
    _thread.start()
    return True

def stop():
    global _running
    _running = False

def get_frame() -> bytes:
    with _lock:
        return _latest_frame or _make_placeholder("System Ready")

def get_status() -> dict:
    return {
        "running": _running,
        "source": _current_source,
        "stats": _stats
    }
