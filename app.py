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

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room

# ─── Create App ───────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "falantir-change-this-in-production")

# ─── CORS ─────────────────────────────────────────────────
# Parse CORS_ORIGINS from env. Support:
#   "*"                                    → allow everything (dev)
#   "http://a.com,http://b.com"            → explicit allowlist
#   unset                                  → sensible local defaults
_raw_origins = os.getenv("CORS_ORIGINS", "").strip()
if _raw_origins == "" or _raw_origins == "*":
    _cors_origins = "*"
    _cors_origins_set = None
else:
    _cors_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    _cors_origins_set = set(_cors_origins)

print(f"CORS: allowed origins = {_cors_origins}")

CORS(
    app,
    origins=_cors_origins,
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)

# Async mode:
#   • Local dev (python app.py)      → "threading"  (default)
#   • Production (gunicorn on Railway)→ "gevent"     (set SOCKETIO_ASYNC_MODE=gevent)
# The gunicorn worker is geventwebsocket.gunicorn.workers.GeventWebSocketWorker,
# which monkey-patches gevent so pymongo / requests run cooperatively.
_async_mode = os.getenv("SOCKETIO_ASYNC_MODE", "threading")
print(f"SocketIO async_mode = {_async_mode}")

socketio = SocketIO(
    app,
    cors_allowed_origins=_cors_origins,
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


# ─── Force CORS on ALL responses (incl. 500 errors) ─────
# Flask-CORS skips error responses by default — so a 500 appears to
# the browser as a CORS error even though it's really a server error.
#
# Logic:
#   - If CORS_ORIGINS is "*" (or unset), echo any Origin header we see.
#   - Otherwise, only echo Origin when it's in our allowlist.
#   - Always set the other CORS headers so preflights work.
@app.after_request
def _force_cors(response):
    origin = request.headers.get("Origin", "")

    if origin:
        if _cors_origins == "*" or (_cors_origins_set and origin in _cors_origins_set):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Credentials"] = "true"

    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    )
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, X-Requested-With"
    )
    return response


# ─── Global OPTIONS handler (CORS Preflight) ─────────────
@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    from flask import make_response
    response = make_response()
    origin = request.headers.get("Origin", "")
    # Only reflect allowlisted origins (required when credentials are allowed —
    # the browser rejects "*" + credentials). Set CORS_ORIGINS on Railway.
    if origin and (_cors_origins == "*" or (_cors_origins_set and origin in _cors_origins_set)):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response, 200


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

import threading

def _warmup_models():
    """Load/warm AI models off the request path. Failures are non-fatal."""
    try:
        from api.services.inference_pipeline import load_all_models
        load_all_models()
        print("MODELS: warmup complete")
    except Exception as model_err:
        print(f"MODEL LOAD WARNING: {model_err}")
        print("  AI vision degraded; auth/user routes still work normally.")


def init_app(load_models=True):
    """Initialize app: DB first (critical), then AI models (optional)."""
    print("=" * 50)
    print("  FALANTIR v2 — Autonomous AI Security Agent")
    print("=" * 50)

    # Critical: database must connect
    from api.database_v2 import init_db
    init_db()

    if load_models:
        _warmup_models()

    print("=" * 50)
    print("  System ready")
    print("=" * 50)


# ─── One-time lazy init on first request ─────────────────
# Keeps the server booting instantly (no 502): DB connects on the first REAL
# request, and AI models warm up in a BACKGROUND thread so they never block a
# request or the platform health check. Thread-safe via a lock.
_app_initialized = False
_init_lock = threading.Lock()
# Paths that must always return fast — never trigger init (health/preflight).
_SKIP_INIT_PATHS = {"/api/health", "/"}

@app.before_request
def _lazy_init():
    global _app_initialized
    if request.method == "OPTIONS" or request.path in _SKIP_INIT_PATHS:
        return
    if _app_initialized:
        return
    with _init_lock:
        if _app_initialized:
            return
        _app_initialized = True
        try:
            init_app(load_models=False)          # DB now (fast)
        except Exception as err:
            print(f"STARTUP ERROR: {err}")
            print("  Some routes may fail until DB is reachable.")
        # Warm AI models in the background — first request isn't blocked.
        threading.Thread(target=_warmup_models, daemon=True).start()

# Alias for gunicorn: `gunicorn app:app`
application = app


if __name__ == "__main__":
    # Local dev only. In production gunicorn imports `app:app`; this block
    # does NOT run, so the gunicorn start command controls host/port/worker.
    init_app()
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 8000))
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=debug,
        allow_unsafe_werkzeug=True,
    )
