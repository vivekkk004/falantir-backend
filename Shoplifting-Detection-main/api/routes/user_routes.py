from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from bson import ObjectId
from api.database import users_collection
from api.schemas import UserResponse, UserUpdateRequest, user_to_response
from api.auth import get_current_user

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return user_to_response(current_user)


@router.put("/me", response_model=UserResponse)
async def update_me(
    request: UserUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    update_data = {}

    if request.name is not None:
        update_data["name"] = request.name

    if request.email is not None:
        existing = await users_collection.find_one({
            "email": request.email,
            "_id": {"$ne": ObjectId(current_user["_id"])},
        })
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use",
            )
        update_data["email"] = request.email

    if not update_data:
        return user_to_response(current_user)

    await users_collection.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": update_data},
    )

    updated_user = await users_collection.find_one({"_id": ObjectId(current_user["_id"])})
    return user_to_response(updated_user)


@router.get("/", response_model=List[UserResponse])
async def get_all_users(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    users = []
    async for user in users_collection.find():
        users.append(user_to_response(user))
    return users


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: str,
    current_user: dict = Depends(get_current_user),
):
    if not ObjectId.is_valid(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID",
        )

    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user_to_response(user)
