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
        import base64
        from api.services.inference_pipeline import analyze_frame

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"success": False, "data": None, "error": "Cannot open video file"}), 400

        total_frames = 0
        frames_analyzed = 0
        peak_result = None
        peak_threat = -1
        peak_frame = None
        all_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            # Analyze every 5th frame (more thorough than every 10th)
            if total_frames % 5 != 0:
                continue

            frames_analyzed += 1

            # Run Gemini every 20 analyzed frames to stay under rate limits
            run_gemini = (frames_analyzed % 20 == 0)
            result = analyze_frame(frame, run_gemini=run_gemini)

            # Collect all non-safe detections
            if result["threat_level"] > 0:
                all_detections.append({
                    "frame": total_frames,
                    "threat_label": result["threat_label"],
                    "confidence": result["confidence"],
                    "yolo_objects": result["yolo_objects"],
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
                    "gemini_description": result["gemini_description"],
                    "yolo_objects": result["yolo_objects"],
                    "inference_time_ms": result["inference_time_ms"],
                }

        cap.release()

        if peak_result is None:
            return jsonify({"success": False, "data": None, "error": "No frames could be analyzed"}), 400

        # Always run Gemini on the peak frame to get a real description
        if peak_frame is not None and (
            not peak_result.get("gemini_description")
            or peak_result["gemini_description"] == "Skipped (frame skip)"
        ):
            final = analyze_frame(peak_frame, run_gemini=True)
            peak_result["gemini_description"] = final["gemini_description"]
            # Also update YOLO if it found more objects
            if len(final["yolo_objects"]) > len(peak_result.get("yolo_objects", [])):
                peak_result["yolo_objects"] = final["yolo_objects"]

        peak_result["total_frames"] = total_frames
        peak_result["frames_analyzed"] = frames_analyzed
        peak_result["threat_detections"] = len(all_detections)

        # Save to database if threat detected
        if peak_result["threat_level"] > 0 and peak_frame is not None:
            _, snap_buf = cv2.imencode(".jpg", peak_frame)
            incident = {
                "agent_id": "video_upload",
                "threat_label": peak_result["threat_label"],
                "threat_level": peak_result["threat_level"],
                "confidence": peak_result["confidence"],
                "gemini_description": peak_result["gemini_description"],
                "yolo_objects": peak_result["yolo_objects"],
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
