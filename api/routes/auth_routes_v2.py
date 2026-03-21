"""Falantir v2 — Auth routes (Flask Blueprint)."""

from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from api.database_v2 import users_col
from api.auth_v2 import hash_password, verify_password, create_access_token, user_to_dict

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "data": None, "error": "Request body required"}), 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    phone = data.get("phone")

    if not name or not email or not password:
        return jsonify({"success": False, "data": None, "error": "Name, email, and password are required"}), 400

    if len(password) < 6:
        return jsonify({"success": False, "data": None, "error": "Password must be at least 6 characters"}), 400

    # Check existing
    if users_col().find_one({"email": email}):
        return jsonify({"success": False, "data": None, "error": "Email already registered"}), 400

    user_doc = {
        "name": name,
        "email": email,
        "phone": phone,
        "hashed_password": hash_password(password),
        "role": "user",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }
    result = users_col().insert_one(user_doc)
    user_doc["_id"] = result.inserted_id

    token = create_access_token(data={"sub": str(user_doc["_id"])})

    return jsonify({
        "success": True,
        "data": {
            "token": token,
            "token_type": "bearer",
            "user": user_to_dict(user_doc),
        },
        "error": None,
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "data": None, "error": "Request body required"}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = users_col().find_one({"email": email})
    if not user or not verify_password(password, user["hashed_password"]):
        return jsonify({"success": False, "data": None, "error": "Invalid email or password"}), 401

    if not user.get("is_active", True):
        return jsonify({"success": False, "data": None, "error": "Account deactivated"}), 403

    token = create_access_token(data={"sub": str(user["_id"])})

    return jsonify({
        "success": True,
        "data": {
            "token": token,
            "token_type": "bearer",
            "user": user_to_dict(user),
        },
        "error": None,
    })


@auth_bp.route("/logout", methods=["POST"])
def logout():
    return jsonify({"success": True, "data": {"message": "Logged out successfully"}, "error": None})
