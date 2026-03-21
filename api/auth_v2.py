"""
Falantir v2 — Authentication helpers for Flask.

JWT tokens with bcrypt password hashing.
"""

import os
from datetime import datetime, timedelta, timezone
from functools import wraps
import bcrypt
from jose import JWTError, jwt
from flask import request, jsonify, g
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "falantir-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# ─── Password Hashing ────────────────────────────────────

def hash_password(password):
    pw_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password, hashed_password):
    pw_bytes = plain_password.encode("utf-8")[:72]
    return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))


# ─── JWT ──────────────────────────────────────────────────

def create_access_token(data, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ─── Flask Auth Decorator ────────────────────────────────

def login_required(f):
    """Decorator that validates JWT and injects current user into g.user."""
    @wraps(f)
    def decorated(*args, **kwargs):
        from api.database_v2 import users_col
        from bson import ObjectId

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"success": False, "data": None, "error": "Missing or invalid token"}), 401

        token = auth_header.split(" ", 1)[1]
        payload = decode_token(token)
        if payload is None:
            return jsonify({"success": False, "data": None, "error": "Invalid or expired token"}), 401

        user_id = payload.get("sub")
        if not user_id:
            return jsonify({"success": False, "data": None, "error": "Invalid token payload"}), 401

        user = users_col().find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"success": False, "data": None, "error": "User not found"}), 401

        if not user.get("is_active", True):
            return jsonify({"success": False, "data": None, "error": "Account deactivated"}), 403

        user["_id"] = str(user["_id"])
        g.user = user
        return f(*args, **kwargs)

    return decorated


def user_to_dict(user):
    """Convert a MongoDB user document to a safe dict."""
    return {
        "id": str(user.get("_id", "")),
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "phone": user.get("phone"),
        "role": user.get("role", "user"),
        "is_active": user.get("is_active", True),
        "created_at": user.get("created_at").isoformat() if user.get("created_at") else None,
    }
