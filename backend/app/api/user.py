from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from ..database import get_db
from ..core.deps import get_current_user
from ..models.user import User

router = APIRouter(prefix="/api/user", tags=["user"])


class UpdateAPIKeyRequest(BaseModel):
    api_key: str


class UpdateProfileRequest(BaseModel):
    email: str


@router.get("/profile")
async def get_profile(user: User = Depends(get_current_user)):
    """Get current user profile."""
    return {
        "id": user.id,
        "email": user.email,
        "created_at": user.created_at,
        "updated_at": user.updated_at
    }


@router.put("/profile")
async def update_profile(
    profile_data: UpdateProfileRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile."""
    # Check if email is already taken by another user
    existing_user = db.query(User).filter(
        User.email == profile_data.email,
        User.id != user.id
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already in use"
        )

    user.email = profile_data.email
    db.commit()
    db.refresh(user)

    return {
        "message": "Profile updated successfully",
        "user": {
            "id": user.id,
            "email": user.email
        }
    }


@router.post("/apikey")
async def update_api_key(
    api_key_data: UpdateAPIKeyRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user's OpenAI API key."""
    # Validate API key format (basic check)
    if api_key_data.api_key and not api_key_data.api_key.startswith("sk-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid API key format. OpenAI API keys should start with 'sk-'"
        )

    # Encrypt and store the API key
    user.set_encrypted_api_key(api_key_data.api_key)
    db.commit()

    return {"message": "API key updated successfully"}


@router.get("/apikey/check")
async def check_api_key(user: User = Depends(get_current_user)):
    """Check if user has an API key configured."""
    return {
        "hasApiKey": bool(user.api_key)
    }


@router.delete("/apikey")
async def delete_api_key(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete user's OpenAI API key."""
    user.set_encrypted_api_key(None)
    db.commit()

    return {"message": "API key deleted successfully"}

