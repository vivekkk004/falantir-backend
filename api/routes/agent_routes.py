"""Falantir v2 — Agent management routes (Flask Blueprint)."""

from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, g, Response
from bson import ObjectId
from api.database_v2 import agents_col
from api.auth_v2 import login_required
from api.services import stream_service

agent_bp = Blueprint("agents", __name__, url_prefix="/api/agents")


def _agent_to_dict(agent):
    """Convert MongoDB agent doc to JSON-safe dict."""
    return {
        "id": str(agent["_id"]),
        "name": agent.get("name", ""),
        "location": agent.get("location", ""),
        "camera_uri": agent.get("camera_uri", ""),
        "status": agent.get("status", "stopped"),
        "created_at": agent["created_at"].isoformat() if agent.get("created_at") else None,
    }


@agent_bp.route("/", methods=["GET"])
@login_required
def list_agents():
    agents = [_agent_to_dict(a) for a in agents_col().find().sort("created_at", -1)]
    return jsonify({"success": True, "data": agents, "error": None})


@agent_bp.route("/", methods=["POST"])
@login_required
def create_agent():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "data": None, "error": "Request body required"}), 400

    name = data.get("name", "").strip()
    location = data.get("location", "").strip()
    camera_uri = data.get("camera_uri", "").strip()

    if not name or not camera_uri:
        return jsonify({"success": False, "data": None, "error": "Name and camera_uri are required"}), 400

    agent_doc = {
        "name": name,
        "location": location,
        "camera_uri": camera_uri,
        "status": "stopped",
        "created_at": datetime.now(timezone.utc),
    }
    result = agents_col().insert_one(agent_doc)
    agent_doc["_id"] = result.inserted_id

    return jsonify({"success": True, "data": _agent_to_dict(agent_doc), "error": None}), 201


@agent_bp.route("/<agent_id>", methods=["DELETE"])
@login_required
def delete_agent(agent_id):
    if not ObjectId.is_valid(agent_id):
        return jsonify({"success": False, "data": None, "error": "Invalid agent ID"}), 400

    # Stop stream if running
    stream_service.stop_stream(agent_id)

    result = agents_col().delete_one({"_id": ObjectId(agent_id)})
    if result.deleted_count == 0:
        return jsonify({"success": False, "data": None, "error": "Agent not found"}), 404

    return jsonify({"success": True, "data": {"message": "Agent deleted"}, "error": None})


@agent_bp.route("/<agent_id>/start", methods=["POST"])
@login_required
def start_agent_stream(agent_id):
    if not ObjectId.is_valid(agent_id):
        return jsonify({"success": False, "data": None, "error": "Invalid agent ID"}), 400

    agent = agents_col().find_one({"_id": ObjectId(agent_id)})
    if not agent:
        return jsonify({"success": False, "data": None, "error": "Agent not found"}), 404

    from api.database_v2 import save_incident
    from flask import current_app
    socketio = current_app.extensions.get("socketio")

    result = stream_service.start_stream(
        agent_id=agent_id,
        camera_uri=agent["camera_uri"],
        socketio=socketio,
        db_save_fn=save_incident,
    )

    if result is None:
        return jsonify({"success": False, "data": None, "error": "Maximum cameras reached"}), 429
    if result is False:
        return jsonify({"success": False, "data": None, "error": "Agent already streaming"}), 409

    agents_col().update_one({"_id": ObjectId(agent_id)}, {"$set": {"status": "streaming"}})

    return jsonify({"success": True, "data": {"message": "Stream started", "agent_id": agent_id}, "error": None})


@agent_bp.route("/<agent_id>/stop", methods=["POST"])
@login_required
def stop_agent_stream(agent_id):
    stream_service.stop_stream(agent_id)

    if ObjectId.is_valid(agent_id):
        agents_col().update_one({"_id": ObjectId(agent_id)}, {"$set": {"status": "stopped"}})

    return jsonify({"success": True, "data": {"message": "Stream stopped", "agent_id": agent_id}, "error": None})


@agent_bp.route("/<agent_id>/stream", methods=["GET"])
def agent_video_stream(agent_id):
    """MJPEG video stream for a specific agent — no auth required for <img> tags."""
    import time

    def generate():
        while True:
            frame = stream_service.get_frame(agent_id)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.05)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@agent_bp.route("/<agent_id>/status", methods=["GET"])
@login_required
def agent_status(agent_id):
    status = stream_service.get_agent_status(agent_id)
    return jsonify({"success": True, "data": status, "error": None})
