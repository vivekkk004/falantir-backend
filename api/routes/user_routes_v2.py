"""Falantir v2 — User routes (Flask Blueprint)."""

from flask import Blueprint, request, jsonify, g
from bson import ObjectId
from api.database_v2 import users_col
from api.auth_v2 import login_required, user_to_dict

user_bp = Blueprint("users", __name__, url_prefix="/api/users")


@user_bp.route("/me", methods=["GET"])
@login_required
def get_me():
    return jsonify({"success": True, "data": user_to_dict(g.user), "error": None})


@user_bp.route("/me", methods=["PUT"])
@login_required
def update_me():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "data": None, "error": "Request body required"}), 400

    update = {}
    if "name" in data and data["name"]:
        update["name"] = data["name"].strip()
    if "email" in data and data["email"]:
        new_email = data["email"].strip().lower()
        existing = users_col().find_one({"email": new_email, "_id": {"$ne": ObjectId(g.user["_id"])}})
        if existing:
            return jsonify({"success": False, "data": None, "error": "Email already in use"}), 400
        update["email"] = new_email
    if "phone" in data:
        update["phone"] = data["phone"]

    if not update:
        return jsonify({"success": True, "data": user_to_dict(g.user), "error": None})

    users_col().update_one({"_id": ObjectId(g.user["_id"])}, {"$set": update})
    updated = users_col().find_one({"_id": ObjectId(g.user["_id"])})

    return jsonify({"success": True, "data": user_to_dict(updated), "error": None})


@user_bp.route("/", methods=["GET"])
@login_required
def get_all_users():
    if g.user.get("role") != "admin":
        return jsonify({"success": False, "data": None, "error": "Admin access required"}), 403

    users = [user_to_dict(u) for u in users_col().find()]
    return jsonify({"success": True, "data": users, "error": None})


@user_bp.route("/<user_id>", methods=["GET"])
@login_required
def get_user(user_id):
    if not ObjectId.is_valid(user_id):
        return jsonify({"success": False, "data": None, "error": "Invalid user ID"}), 400

    user = users_col().find_one({"_id": ObjectId(user_id)})
    if not user:
        return jsonify({"success": False, "data": None, "error": "User not found"}), 404

    return jsonify({"success": True, "data": user_to_dict(user), "error": None})
