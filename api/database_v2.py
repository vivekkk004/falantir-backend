"""
Falantir v2 — MongoDB database layer.

Collections: users, agents, incidents, analytics, rl_feedback
Uses PyMongo (sync) for Flask compatibility.
"""

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "falantir"

_client = None
_db = None


def get_db():
    """Get the database instance, connecting if needed.

    TLS is only enabled for remote / Atlas connections. Local dev instances
    (mongodb://localhost/, 127.0.0.1/, plain hostnames) use an unencrypted
    connection, because forcing TLS on a non-TLS local mongod crashes the
    driver with 'SSL handshake failed'.
    """
    global _client, _db
    if _db is None:
        uri_lower = MONGO_URI.lower()
        is_local = (
            "localhost" in uri_lower
            or "127.0.0.1" in uri_lower
            or "host.docker.internal" in uri_lower
        )
        is_srv = uri_lower.startswith("mongodb+srv://")
        use_tls = is_srv or (not is_local)

        kwargs = {"serverSelectionTimeoutMS": 5000}
        if use_tls:
            import certifi
            kwargs["tlsCAFile"] = certifi.where()

        try:
            _client = MongoClient(MONGO_URI, **kwargs)
            _db = _client[DATABASE_NAME]
            print(
                f"DATABASE: Client created for {DATABASE_NAME} "
                f"(tls={'on' if use_tls else 'off'})"
            )
        except Exception as e:
            print(f"DATABASE CONNECTION ERROR: {e}")
            raise e
    return _db


def init_db():
    """Create indexes on startup."""
    db = get_db()

    # Users
    db.users.create_index("email", unique=True)

    # Agents
    db.agents.create_index("name")
    db.agents.create_index("created_at")

    # Incidents
    db.incidents.create_index([("timestamp", DESCENDING)])
    db.incidents.create_index("agent_id")
    db.incidents.create_index("threat_label")

    # Analytics
    db.analytics.create_index([("agent_id", ASCENDING), ("date", ASCENDING)], unique=True)

    # RL Feedback
    db.rl_feedback.create_index("incident_id")
    db.rl_feedback.create_index([("timestamp", DESCENDING)])

    print(f"DATABASE: Connected to {MONGO_URI} — db: {DATABASE_NAME}")


def close_db():
    """Close MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None


# ─── Collection Accessors ─────────────────────────────────

def users_col():
    return get_db().users

def agents_col():
    return get_db().agents

def incidents_col():
    return get_db().incidents

def analytics_col():
    return get_db().analytics

def rl_feedback_col():
    return get_db().rl_feedback


# ─── Incident Helpers ─────────────────────────────────────

def save_incident(incident_data):
    """Save an incident to MongoDB. Called from background thread."""
    try:
        if "timestamp" not in incident_data:
            incident_data["timestamp"] = datetime.now(timezone.utc)
        incidents_col().insert_one(incident_data)
    except Exception as e:
        print(f"DB ERROR (save_incident): {e}")


def update_daily_analytics(agent_id, threat_label):
    """Increment daily analytics counters for an agent."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        analytics_col().update_one(
            {"agent_id": agent_id, "date": today},
            {
                "$inc": {
                    f"counts.{threat_label}": 1,
                    "counts.total": 1,
                },
                "$set": {"updated_at": datetime.now(timezone.utc)},
                "$setOnInsert": {
                    "agent_id": agent_id,
                    "date": today,
                    "timestamp": datetime.now(timezone.utc),
                },
            },
            upsert=True,
        )
    except Exception as e:
        print(f"DB ERROR (analytics): {e}")
