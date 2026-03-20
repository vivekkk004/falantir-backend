"""
MongoDB Document Schemas (reference only — MongoDB is schema-less)

Collection: users
{
    "_id": ObjectId,
    "name": str,
    "email": str (unique),
    "hashed_password": str,
    "role": str,       # "user" | "admin"
    "is_active": bool,
    "created_at": datetime,
}

Collection: alerts
{
    "_id": ObjectId,
    "camera_id": str,
    "camera_label": str,
    "status": str,     # "Shoplifting" | "Not Shoplifting"
    "confidence": float,
    "timestamp": datetime,
    "snapshot": str,   # base64-encoded JPEG or None
}
"""
