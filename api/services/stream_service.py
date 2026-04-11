"""
Multi-Agent Live Stream Service — Falantir v2.1

Each camera agent runs its own background thread, reading frames from an
RTSP stream, webcam, or video file.

v2.1 changes vs v2:
  - YOLO gone; single Gemini 2.5 Flash structured-output call via
    vision_provider abstraction.
  - Motion gate (OpenCV MOG2) in front of the vision provider, so we
    don't burn API credit on empty frames.
  - Idle frames still stream to the browser (annotated with the last
    known threat label) so the live feed never freezes.
"""

import threading
import time
import base64
import os
import cv2
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

from api.services.motion_detector import MotionDetector

load_dotenv()

WIDTH = 640
MAX_CAMERAS = int(os.getenv("MAX_CAMERAS", "5"))
THREAT_THRESHOLD = float(os.getenv("THREAT_THRESHOLD", "0.6"))

# Minimum seconds between two vision-provider calls per agent, even if motion
# stays high. Caps cost for agents pointed at busy scenes.
MIN_ANALYSIS_INTERVAL_S = float(os.getenv("MIN_ANALYSIS_INTERVAL_S", "2.0"))

# How often to emit a lightweight frame to the browser when nothing is being
# analyzed (keeps the live view smooth).
IDLE_FRAME_EMIT_EVERY_N = int(os.getenv("IDLE_FRAME_EMIT_EVERY_N", "5"))


# ─── Agent Registry ───────────────────────────────────────

_agents_lock = threading.Lock()
_active_agents = {}  # agent_id -> {thread, running, latest_frame, latest_result}


def _encode_jpeg(frame, quality=80):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _make_placeholder(msg="Connecting..."):
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, msg, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    return _encode_jpeg(img)


def _overlay_status(frame, text, color=(200, 200, 200)):
    """Small status pill in the top-right corner of the live frame."""
    annotated = frame.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = annotated.shape[1] - tw - 16, 10
    cv2.rectangle(annotated, (x - 4, y), (x + tw + 4, y + th + 8), (30, 30, 30), -1)
    cv2.putText(annotated, text, (x, y + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def _stream_loop(agent_id, camera_uri, socketio, db_save_fn):
    """Background thread for a single camera agent."""
    from api.services.inference_pipeline import analyze_frame

    src = int(camera_uri) if str(camera_uri).isdigit() else camera_uri
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        with _agents_lock:
            if agent_id in _active_agents:
                _active_agents[agent_id]["latest_frame"] = _make_placeholder("Source not found")
                _active_agents[agent_id]["running"] = False
        return

    motion_detector = MotionDetector()
    frame_count = 0
    analyzed_count = 0
    skipped_count = 0
    last_analysis_ts = 0.0
    last_result = None
    reconnect_attempts = 0
    max_reconnect = 5

    while True:
        with _agents_lock:
            if agent_id not in _active_agents or not _active_agents[agent_id]["running"]:
                break

        ret, frame = cap.read()

        if not ret:
            # For video files, loop back to start
            if not str(camera_uri).isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # For RTSP/cameras, try reconnecting
            reconnect_attempts += 1
            if reconnect_attempts > max_reconnect:
                break
            print(f"STREAM [{agent_id}]: Reconnecting... attempt {reconnect_attempts}")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(src)
            continue

        reconnect_attempts = 0
        frame_count += 1

        # Resize for downstream processing
        h, w = frame.shape[:2]
        ratio = WIDTH / float(w)
        frame = cv2.resize(frame, (WIDTH, int(h * ratio)))

        # ─── Motion gate ─────────────────────────────────
        has_motion, motion_ratio = motion_detector.check(frame)
        now = time.time()
        interval_ok = (now - last_analysis_ts) >= MIN_ANALYSIS_INTERVAL_S
        should_analyze = has_motion and interval_ok

        if should_analyze:
            # ─── Expensive path: call vision provider ────
            result = analyze_frame(frame)
            last_analysis_ts = now
            analyzed_count += 1

            annotated = result.pop("annotated_frame", frame)
            frame_jpeg = _encode_jpeg(annotated)
            last_result = result

            with _agents_lock:
                if agent_id in _active_agents:
                    _active_agents[agent_id]["latest_frame"] = frame_jpeg
                    _active_agents[agent_id]["latest_result"] = result
                    _active_agents[agent_id]["frame_count"] = frame_count
                    _active_agents[agent_id]["analyzed_count"] = analyzed_count
                    _active_agents[agent_id]["skipped_count"] = skipped_count

            # Emit agent_update via WebSocket
            update_payload = {
                "agent_id": agent_id,
                "threat_label": result["threat_label"],
                "threat_level": result["threat_level"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "scene_description": result.get("scene_description", ""),
                "gemini_description": result.get("scene_description", ""),  # legacy
                "reasoning": result.get("reasoning", ""),
                "detected_objects": result.get("detected_objects", []),
                "yolo_objects": result.get("detected_objects", []),  # legacy
                "provider_used": result.get("provider_used", "unknown"),
                "model": result.get("model", "unknown"),
                "inference_time_ms": result.get("inference_time_ms", 0),
                "frame_count": frame_count,
                "analyzed_count": analyzed_count,
                "motion_ratio": round(motion_ratio, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if socketio:
                socketio.emit("agent_update", update_payload, room=agent_id)

            # Save incident if suspicious or critical
            if result["threat_level"] >= 1 and result["confidence"] >= THREAT_THRESHOLD:
                _, snap_buf = cv2.imencode(".jpg", annotated)
                incident = {
                    "agent_id": agent_id,
                    "threat_label": result["threat_label"],
                    "threat_level": result["threat_level"],
                    "confidence": result["confidence"],
                    "scene_description": result.get("scene_description", ""),
                    "gemini_description": result.get("scene_description", ""),  # legacy
                    "reasoning": result.get("reasoning", ""),
                    "detected_objects": result.get("detected_objects", []),
                    "yolo_objects": result.get("detected_objects", []),  # legacy
                    "provider_used": result.get("provider_used", "unknown"),
                    "model": result.get("model", "unknown"),
                    "timestamp": datetime.now(timezone.utc),
                    "snapshot": base64.b64encode(snap_buf).decode(),
                    "acknowledged": False,
                }

                if db_save_fn:
                    threading.Thread(target=db_save_fn, args=(incident,), daemon=True).start()

                # Emit critical alert event
                if result["threat_level"] == 2:
                    if socketio:
                        socketio.emit("incident_alert", {
                            "agent_id": agent_id,
                            "threat_label": result["threat_label"],
                            "confidence": result["confidence"],
                            "description": result.get("scene_description", ""),
                            "reasoning": result.get("reasoning", ""),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

        else:
            # ─── Cheap path: no vision call, just keep the stream alive ──
            skipped_count += 1

            # Only encode every Nth idle frame — no need to stream 30 FPS of
            # empty-shop footage, and it keeps CPU low.
            if skipped_count % IDLE_FRAME_EMIT_EVERY_N == 0:
                status_text = "IDLE (no motion)" if not has_motion else "COOLDOWN"
                status_color = (120, 120, 120) if not has_motion else (0, 165, 255)

                # Show the last known threat overlay so the UI isn't blank
                if last_result:
                    lbl = last_result.get("threat_label", "safe").upper()
                    conf = last_result.get("confidence", 0.0)
                    color_map = {
                        "SAFE": (0, 200, 0),
                        "SUSPICIOUS": (0, 165, 255),
                        "CRITICAL": (0, 0, 255),
                    }
                    overlay_color = color_map.get(lbl, (200, 200, 200))
                    cv2.putText(frame, f"{lbl} ({conf:.0%})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2, cv2.LINE_AA)

                frame = _overlay_status(frame, status_text, status_color)
                frame_jpeg = _encode_jpeg(frame)

                with _agents_lock:
                    if agent_id in _active_agents:
                        _active_agents[agent_id]["latest_frame"] = frame_jpeg
                        _active_agents[agent_id]["frame_count"] = frame_count
                        _active_agents[agent_id]["skipped_count"] = skipped_count

        time.sleep(0.01)

    cap.release()
    with _agents_lock:
        if agent_id in _active_agents:
            _active_agents[agent_id]["running"] = False

    # Emit status change
    if socketio:
        socketio.emit("agent_status_change", {
            "agent_id": agent_id,
            "status": "stopped",
        })


# ─── Public API ───────────────────────────────────────────

def start_stream(agent_id, camera_uri, socketio=None, db_save_fn=None):
    """Start a live stream for an agent."""
    with _agents_lock:
        if agent_id in _active_agents and _active_agents[agent_id]["running"]:
            return False  # Already running

        if len([a for a in _active_agents.values() if a["running"]]) >= MAX_CAMERAS:
            return None  # Max cameras reached

        _active_agents[agent_id] = {
            "running": True,
            "latest_frame": _make_placeholder("Starting..."),
            "latest_result": None,
            "frame_count": 0,
            "analyzed_count": 0,
            "skipped_count": 0,
            "thread": None,
        }

    thread = threading.Thread(
        target=_stream_loop,
        args=(agent_id, camera_uri, socketio, db_save_fn),
        daemon=True,
    )
    thread.start()

    with _agents_lock:
        _active_agents[agent_id]["thread"] = thread

    if socketio:
        socketio.emit("agent_status_change", {
            "agent_id": agent_id,
            "status": "streaming",
        })

    return True


def stop_stream(agent_id):
    """Stop streaming for an agent."""
    with _agents_lock:
        if agent_id in _active_agents:
            _active_agents[agent_id]["running"] = False
            return True
    return False


def get_frame(agent_id):
    """Get the latest JPEG frame for an agent."""
    with _agents_lock:
        agent = _active_agents.get(agent_id)
        if agent:
            return agent["latest_frame"]
    return _make_placeholder("Agent not found")


def get_agent_status(agent_id):
    """Get streaming status for a specific agent."""
    with _agents_lock:
        agent = _active_agents.get(agent_id)
        if agent:
            return {
                "running": agent["running"],
                "frame_count": agent.get("frame_count", 0),
                "analyzed_count": agent.get("analyzed_count", 0),
                "skipped_count": agent.get("skipped_count", 0),
                "latest_result": agent.get("latest_result"),
            }
    return {
        "running": False, "frame_count": 0,
        "analyzed_count": 0, "skipped_count": 0, "latest_result": None,
    }


def get_all_statuses():
    """Return status of all active agents."""
    with _agents_lock:
        return {
            aid: {
                "running": a["running"],
                "frame_count": a.get("frame_count", 0),
                "analyzed_count": a.get("analyzed_count", 0),
                "skipped_count": a.get("skipped_count", 0),
            }
            for aid, a in _active_agents.items()
        }


def stop_all():
    """Stop all running streams."""
    with _agents_lock:
        for agent in _active_agents.values():
            agent["running"] = False
