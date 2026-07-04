from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Literal, Dict
from ..database import get_db
from ..core.deps import get_current_user
from ..models.user import User

router = APIRouter(prefix="/api/user", tags=["user"])

# Allowed providers. Keep in sync with services/llm/provider.Provider.
ProviderName = Literal["openai", "anthropic", "ollama"]


class UpdateAPIKeyRequest(BaseModel):
    api_key: str


class UpdateProfileRequest(BaseModel):
    email: str


# Per-provider API-key format hints for validation.
# Ollama Cloud keys have no public stable prefix, so we only require non-empty.
_PROVIDER_KEY_VALIDATION: Dict[str, dict] = {
    "openai": {
        "prefix": "sk-",
        "message": "OpenAI API keys should start with 'sk-'",
    },
    "anthropic": {
        "prefix": "sk-ant-",
        "message": "Anthropic API keys should start with 'sk-ant-'",
    },
    "ollama": {
        "prefix": None,  # no validation beyond non-empty
        "message": "Invalid Ollama Cloud API key",
    },
}


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


@router.post("/apikey/{provider}")
async def update_api_key(
    provider: ProviderName,
    api_key_data: UpdateAPIKeyRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update the user's API key for a specific provider."""
    rules = _PROVIDER_KEY_VALIDATION[provider]
    key = api_key_data.api_key
    if key:
        prefix = rules["prefix"]
        if prefix and not key.startswith(prefix):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid API key format. {rules['message']}"
            )

    user.set_encrypted_key(provider, key)
    db.commit()

    return {"message": f"{provider} API key updated successfully"}


@router.get("/apikey/check")
async def check_api_key(user: User = Depends(get_current_user)):
    """Check which providers the user has API keys configured for."""
    return {
        "openai": bool(user.openai_api_key or user.api_key),
        "anthropic": bool(user.anthropic_api_key),
        "ollama": bool(user.ollama_api_key),
    }


@router.delete("/apikey/{provider}")
async def delete_api_key(
    provider: ProviderName,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete the user's API key for a specific provider."""
    user.set_encrypted_key(provider, None)
    db.commit()

    return {"message": f"{provider} API key deleted successfully"}
