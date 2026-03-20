import traceback
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from api.database import users_collection
from api.schemas import RegisterRequest, LoginRequest, TokenResponse, MessageResponse, user_to_response
from api.auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    try:
        # Check if email already exists
        existing_user = await users_collection.find_one({"email": request.email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Create user document
        user_doc = {
            "name": request.name,
            "email": request.email,
            "phone": request.phone,
            "hashed_password": hash_password(request.password),
            "role": "user",
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }
        result = await users_collection.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id

        # Generate token
        token = create_access_token(data={"sub": str(user_doc["_id"])})

        user_data = user_to_response(user_doc)
        return {
            "token": token,
            "token_type": "bearer",
            "user": user_data.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as e:
        print("=== REGISTER ERROR ===")
        traceback.print_exc()
        print("=====================")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    # Find user
    user = await users_collection.find_one({"email": request.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Verify password
    if not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    # Generate token
    token = create_access_token(data={"sub": str(user["_id"])})

    return TokenResponse(
        token=token,
        user=user_to_response(user),
    )


@router.post("/logout", response_model=MessageResponse)
async def logout():
    return MessageResponse(message="Logged out successfully")
