from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from bson import ObjectId


# ─── Auth Schemas ──────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    user: "UserResponse"


# ─── User Schemas ──────────────────────────────────────────

class UserResponse(BaseModel):
    id: Optional[str] = None
    name: str
    email: str
    phone: Optional[str] = None
    role: str = "user"
    is_active: bool = True
    created_at: Optional[datetime] = None

    class Config:
        populate_by_name = True


def user_to_response(user: dict) -> "UserResponse":
    """Convert a MongoDB user document to UserResponse."""
    return UserResponse(
        id=str(user.get("_id", "")),
        name=user.get("name", ""),
        email=user.get("email", ""),
        phone=user.get("phone"),
        role=user.get("role", "user"),
        is_active=user.get("is_active", True),
        created_at=user.get("created_at"),
    )


class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None


class MessageResponse(BaseModel):
    message: str


# Rebuild forward ref
TokenResponse.model_rebuild()
