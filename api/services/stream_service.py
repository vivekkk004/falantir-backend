"""
Multi-Agent Live Stream Service.

Each camera agent runs its own background thread, reading frames from an
RTSP stream or video file. On every processed frame, all three models run
in parallel and results are emitted over WebSocket.
"""

import threading
import time
import base64
import os
import cv2
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

WIDTH = 640
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))
MAX_CAMERAS = int(os.getenv("MAX_CAMERAS", "5"))
THREAT_THRESHOLD = float(os.getenv("THREAT_THRESHOLD", "0.6"))

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

    frame_count = 0
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

        # Resize
        h, w = frame.shape[:2]
        ratio = WIDTH / float(w)
        frame = cv2.resize(frame, (WIDTH, int(h * ratio)))

        # Run Gemini only every FRAME_SKIP frames (rate limiting)
        run_gemini = (frame_count % FRAME_SKIP == 0)

        # Run all three models in parallel
        result = analyze_frame(frame, run_gemini=run_gemini)

        # Store latest frame and result
        annotated = result.pop("annotated_frame", frame)
        frame_jpeg = _encode_jpeg(annotated)

        with _agents_lock:
            if agent_id in _active_agents:
                _active_agents[agent_id]["latest_frame"] = frame_jpeg
                _active_agents[agent_id]["latest_result"] = result
                _active_agents[agent_id]["frame_count"] = frame_count

        # Emit agent_update via WebSocket
        update_payload = {
            "agent_id": agent_id,
            "threat_label": result["threat_label"],
            "threat_level": result["threat_level"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "gemini_description": result["gemini_description"],
            "yolo_objects": result["yolo_objects"],
            "inference_time_ms": result["inference_time_ms"],
            "frame_count": frame_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if socketio:
            socketio.emit("agent_update", update_payload, room=agent_id)

        # Save incident to DB if suspicious or critical
        if result["threat_level"] >= 1 and result["confidence"] >= THREAT_THRESHOLD:
            if frame_count % 40 == 0:  # Don't save every frame
                _, snap_buf = cv2.imencode(".jpg", annotated)
                incident = {
                    "agent_id": agent_id,
                    "threat_label": result["threat_label"],
                    "threat_level": result["threat_level"],
                    "confidence": result["confidence"],
                    "gemini_description": result["gemini_description"],
                    "yolo_objects": result["yolo_objects"],
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
                            "description": result["gemini_description"],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

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
                "latest_result": agent.get("latest_result"),
            }
    return {"running": False, "frame_count": 0, "latest_result": None}


def get_all_statuses():
    """Return status of all active agents."""
    with _agents_lock:
        return {
            aid: {
                "running": a["running"],
                "frame_count": a.get("frame_count", 0),
            }
            for aid, a in _active_agents.items()
        }


def stop_all():
    """Stop all running streams."""
    with _agents_lock:
        for agent in _active_agents.values():
            agent["running"] = False
