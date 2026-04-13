"""Falantir v2 — Detection & analysis routes (Flask Blueprint)."""

import os
import tempfile
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, g
from api.auth_v2 import login_required
from api.database_v2 import incidents_col, rl_feedback_col, analytics_col, save_incident
from api.services.inference_pipeline import get_models_status
from api.notifications import notify_all
from api.database_v2 import users_col
from bson import ObjectId

detection_bp = Blueprint("detection", __name__, url_prefix="/api/detection")


@detection_bp.route("/models", methods=["GET"])
@login_required
def models_status():
    """Return status of all three AI models."""
    status = get_models_status()
    return jsonify({"success": True, "data": status, "error": None})


@detection_bp.route("/upload", methods=["POST"])
@login_required
def upload_video():
    """Upload a video file, analyze with all three models, return results."""
    if "video" not in request.files:
        return jsonify({"success": False, "data": None, "error": "No video file provided"}), 400

    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"success": False, "data": None, "error": "Empty filename"}), 400

    # Save to temp file
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    video_file.save(tmp_path)
    tmp.close()

    try:
        import cv2
        import time
        import base64
        from api.services.inference_pipeline import analyze_frame

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"success": False, "data": None, "error": "Cannot open video file"}), 400

        # ─── Quota-friendly sampling ──────────────────────────────
        # Gemini 2.5 Flash Lite free tier is 15 RPM. We must throttle to
        # ~13 RPM to stay safely under the limit, and stop the moment the
        # API returns a 429 so we don't burn through the remaining quota.
        TARGET_SAMPLES = int(os.getenv("UPLOAD_TARGET_SAMPLES", "12"))
        MIN_INTERVAL_S = float(os.getenv("UPLOAD_MIN_INTERVAL_S", "4.5"))

        total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        # Space samples EVENLY across the whole video, not just the start.
        stride = max(1, total_frames_count // TARGET_SAMPLES) if total_frames_count else 15

        total_frames = 0
        frames_analyzed = 0
        frames_failed = 0
        peak_result = None
        peak_threat = -1
        peak_frame = None
        all_detections = []
        last_call_ts = 0.0
        quota_hit = False

        def _looks_like_quota_error(r):
            msg = (r.get("reasoning", "") + " " + r.get("scene_description", "")).lower()
            return "429" in msg or "quota" in msg or "rate" in msg

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            if total_frames % stride != 0:
                continue

            if frames_analyzed >= TARGET_SAMPLES or quota_hit:
                # Keep reading to compute accurate total_frames, no more API calls
                continue

            # Throttle between Gemini calls so we stay under 15 RPM
            now = time.time()
            wait = MIN_INTERVAL_S - (now - last_call_ts)
            if wait > 0 and last_call_ts > 0:
                time.sleep(wait)

            result = analyze_frame(frame)
            last_call_ts = time.time()

            # Detect rate limit → stop early
            if _looks_like_quota_error(result):
                quota_hit = True
                frames_failed += 1
                print(f"UPLOAD: Gemini quota hit after {frames_analyzed} calls — stopping")
                continue

            frames_analyzed += 1

            # Collect all non-safe detections
            if result["threat_level"] > 0:
                all_detections.append({
                    "frame": total_frames,
                    "threat_label": result["threat_label"],
                    "confidence": result["confidence"],
                    "detected_objects": result["detected_objects"],
                })

            # Track peak threat frame
            if result["threat_level"] > peak_threat or (
                result["threat_level"] == peak_threat
                and result["confidence"] > (peak_result or {}).get("confidence", 0)
            ):
                peak_threat = result["threat_level"]
                peak_frame = frame.copy()
                peak_result = {
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
                }

        cap.release()

        if frames_analyzed == 0:
            # Every analysis attempt failed — most likely quota exhausted
            err_msg = (
                "Gemini API quota exhausted (free tier is 15 RPM / ~1000 RPD). "
                "Wait ~60s for the per-minute window to reset, or until tomorrow "
                "for the daily quota."
                if quota_hit
                else "No frames could be analyzed"
            )
            return jsonify({
                "success": False,
                "data": None,
                "error": err_msg,
                "quota_hit": quota_hit,
            }), 429 if quota_hit else 400

        if peak_result is None:
            return jsonify({"success": False, "data": None, "error": "No frames could be analyzed"}), 400

        peak_result["total_frames"] = total_frames
        peak_result["frames_analyzed"] = frames_analyzed
        peak_result["frames_failed"] = frames_failed
        peak_result["quota_hit"] = quota_hit
        peak_result["threat_detections"] = len(all_detections)

        # Save to database if threat detected
        if peak_result["threat_level"] > 0 and peak_frame is not None:
            _, snap_buf = cv2.imencode(".jpg", peak_frame)
            incident = {
                "agent_id": "video_upload",
                "threat_label": peak_result["threat_label"],
                "threat_level": peak_result["threat_level"],
                "confidence": peak_result["confidence"],
                "scene_description": peak_result["scene_description"],
                "gemini_description": peak_result["scene_description"],  # legacy
                "reasoning": peak_result["reasoning"],
                "detected_objects": peak_result["detected_objects"],
                "yolo_objects": peak_result["detected_objects"],  # legacy
                "provider_used": peak_result["provider_used"],
                "model": peak_result["model"],
                "timestamp": datetime.now(timezone.utc),
                "snapshot": base64.b64encode(snap_buf).decode(),
                "acknowledged": False,
                "source": "video_upload",
                "filename": video_file.filename,
            }
            save_incident(incident)

        return jsonify({"success": True, "data": peak_result, "error": None})

    finally:
        # Always delete temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@detection_bp.route("/incidents", methods=["GET"])
@login_required
def get_incidents():
    """Return paginated incident history."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    per_page = min(per_page, 100)
    agent_id = request.args.get("agent_id")

    query = {}
    if agent_id:
        query["agent_id"] = agent_id

    total = incidents_col().count_documents(query)
    incidents = list(
        incidents_col()
        .find(query)
        .sort("timestamp", -1)
        .skip((page - 1) * per_page)
        .limit(per_page)
    )

    for inc in incidents:
        inc["_id"] = str(inc["_id"])
        if isinstance(inc.get("timestamp"), datetime):
            inc["timestamp"] = inc["timestamp"].isoformat()

        # Back-compat: old records stored `yolo_objects`; new code expects
        # `detected_objects`. Make both available either way.
        if "detected_objects" not in inc and "yolo_objects" in inc:
            inc["detected_objects"] = inc["yolo_objects"]
        if "yolo_objects" not in inc and "detected_objects" in inc:
            inc["yolo_objects"] = inc["detected_objects"]
        if "scene_description" not in inc and "gemini_description" in inc:
            inc["scene_description"] = inc["gemini_description"]
        if "gemini_description" not in inc and "scene_description" in inc:
            inc["gemini_description"] = inc["scene_description"]

    return jsonify({
        "success": True,
        "data": {
            "incidents": incidents,
            "total": total,
            "page": page,
            "per_page": per_page,
        },
        "error": None,
    })


@detection_bp.route("/incidents/<incident_id>/acknowledge", methods=["POST"])
@login_required
def acknowledge_incident(incident_id):
    """Mark an incident as acknowledged."""
    if not ObjectId.is_valid(incident_id):
        return jsonify({"success": False, "data": None, "error": "Invalid incident ID"}), 400

    result = incidents_col().update_one(
        {"_id": ObjectId(incident_id)},
        {"$set": {"acknowledged": True, "acknowledged_at": datetime.now(timezone.utc)}},
    )

    if result.matched_count == 0:
        return jsonify({"success": False, "data": None, "error": "Incident not found"}), 404

    return jsonify({"success": True, "data": {"message": "Incident acknowledged"}, "error": None})


@detection_bp.route("/feedback", methods=["POST"])
@login_required
def submit_feedback():
    """Submit RL feedback on an incident (correct / false_positive)."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "data": None, "error": "Request body required"}), 400

    incident_id = data.get("incident_id", "")
    verdict = data.get("verdict", "")  # "correct" or "false_positive"
    correct_label = data.get("correct_label")  # If false_positive, what it should be

    if not incident_id or verdict not in ("correct", "false_positive"):
        return jsonify({"success": False, "data": None, "error": "incident_id and verdict (correct/false_positive) required"}), 400

    feedback_doc = {
        "incident_id": incident_id,
        "user_id": g.user["_id"],
        "verdict": verdict,
        "correct_label": correct_label,
        "timestamp": datetime.now(timezone.utc),
    }
    rl_feedback_col().insert_one(feedback_doc)

    return jsonify({"success": True, "data": {"message": "Feedback recorded"}, "error": None})


@detection_bp.route("/stats", methods=["GET"])
@login_required
def get_stats():
    """Return overall analytics stats."""
    total_incidents = incidents_col().count_documents({})
    critical = incidents_col().count_documents({"threat_label": "critical"})
    suspicious = incidents_col().count_documents({"threat_label": "suspicious"})

    # Per-agent stats
    pipeline = [
        {"$group": {
            "_id": "$agent_id",
            "total": {"$sum": 1},
            "avg_confidence": {"$avg": "$confidence"},
            "last_active": {"$max": "$timestamp"},
        }},
    ]
    agent_stats = list(incidents_col().aggregate(pipeline))
    for s in agent_stats:
        s["agent_id"] = s.pop("_id")
        if isinstance(s.get("last_active"), datetime):
            s["last_active"] = s["last_active"].isoformat()
        s["avg_confidence"] = round(s.get("avg_confidence", 0), 4)

    return jsonify({
        "success": True,
        "data": {
            "total_incidents": total_incidents,
            "critical_count": critical,
            "suspicious_count": suspicious,
            "safe_count": total_incidents - critical - suspicious,
            "agent_stats": agent_stats,
        },
        "error": None,
    })


@detection_bp.route("/analytics/daily", methods=["GET"])
@login_required
def daily_analytics():
    """Return daily incident counts for the past 30 days."""
    days = request.args.get("days", 30, type=int)
    days = min(days, 90)

    from datetime import timedelta
    since = datetime.now(timezone.utc) - timedelta(days=days)

    records = list(
        analytics_col()
        .find({"timestamp": {"$gte": since}})
        .sort("date", 1)
    )

    for r in records:
        r["_id"] = str(r["_id"])
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].isoformat()
        if isinstance(r.get("updated_at"), datetime):
            r["updated_at"] = r["updated_at"].isoformat()

    return jsonify({"success": True, "data": records, "error": None})


@detection_bp.route("/alert/manual", methods=["POST"])
@login_required
def manual_alert():
    """Manually trigger an alert — sends Twilio SMS and email to all active users."""
    data = request.get_json() or {}
    message = data.get("message", "Manual security alert triggered from Falantir dashboard.")

    users = list(users_col().find({"is_active": True}))
    results = []
    for user in users:
        r = notify_all(user.get("email"), user.get("phone"), message)
        results.append({"user": user.get("email"), "results": r})

    return jsonify({"success": True, "data": {"notifications_sent": results}, "error": None})
