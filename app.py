"""
Falantir v2 — Main Flask + Flask-SocketIO Application.

This is the entry point for the v2 backend. Run with:
    python app.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room

# ─── Create App ───────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "falantir-change-this-in-production")

CORS(app, origins=os.getenv("CORS_ORIGINS", "*").split(","), supports_credentials=True)

# Use eventlet when running under gunicorn (Render) for proper WebSocket support.
# Fall back to threading for local dev (python app.py).
_async_mode = "eventlet" if os.getenv("SERVER_SOFTWARE", "").startswith("gunicorn") else "threading"

socketio = SocketIO(
    app,
    cors_allowed_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    async_mode=_async_mode,
)

# ─── Register Blueprints ─────────────────────────────────

from api.routes.auth_routes_v2 import auth_bp
from api.routes.user_routes_v2 import user_bp
from api.routes.agent_routes import agent_bp
from api.routes.detection_routes_v2 import detection_bp

app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(agent_bp)
app.register_blueprint(detection_bp)

# ─── Root & Health ────────────────────────────────────────

@app.route("/")
def root():
    return jsonify({"message": "Falantir v2 API is running"})


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy", "version": "2.0.0"})


# ─── WebSocket Events ────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    print("WS: Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("WS: Client disconnected")


@socketio.on("join_agent")
def handle_join_agent(data):
    agent_id = data.get("agent_id") if isinstance(data, dict) else data
    if agent_id:
        join_room(agent_id)
        print(f"WS: Client joined room {agent_id}")


@socketio.on("leave_agent")
def handle_leave_agent(data):
    agent_id = data.get("agent_id") if isinstance(data, dict) else data
    if agent_id:
        leave_room(agent_id)
        print(f"WS: Client left room {agent_id}")

# ─── Startup ─────────────────────────────────────────────

def init_app():
    """Initialize database and load all models on startup."""
    from api.database_v2 import init_db
    from api.services.inference_pipeline import load_all_models

    print("=" * 50)
    print("  FALANTIR v2 — Autonomous AI Security Agent")
    print("=" * 50)

    init_db()
    load_all_models()

    print("=" * 50)
    print("  System ready")
    print("=" * 50)


# ─── Gunicorn entry-point ────────────────────────────────
# Gunicorn never executes the __main__ block, so we initialise here.
# Guard prevents double-init when running via `python app.py`.
if os.getenv("SERVER_SOFTWARE", "").startswith("gunicorn"):
    init_app()

# Alias expected by gunicorn: `gunicorn app:app`
application = app


if __name__ == "__main__":
    init_app()
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    socketio.run(
        app,
        host="0.0.0.0",
        port=8000,
        debug=debug,
        allow_unsafe_werkzeug=True,
    )
