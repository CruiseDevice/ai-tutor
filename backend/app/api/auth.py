from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from sqlalchemy.orm import Session
import logging
from ..database import get_db
from ..schemas.auth import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    PasswordResetRequest,
    PasswordResetConfirm
)
from ..services.auth_service import AuthService
from ..core.deps import get_current_user
from ..core.rate_limit_dep import check_rate_limit
from ..models.user import User
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(
    user_data: UserCreate,
    response: Response,
    request: Request,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    # Apply rate limiting
    await check_rate_limit(request, settings.RATE_LIMIT_AUTH_PER_MINUTE)

    try:
        logger.info(f"Registration attempt for email: {user_data.email}")
        user = AuthService.create_user(db, user_data)

        # Create session
        session, token = AuthService.create_session(db, user.id)

        # Set cookie with secure settings based on environment
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            secure=settings.COOKIE_SECURE or settings.is_production,
            samesite=settings.COOKIE_SAMESITE,
            max_age=7 * 24 * 60 * 60  # 7 days
        )

        logger.info(f"Registration successful for user: {user.id}")
        return TokenResponse(
            message="Registration successful",
            user=UserResponse.model_validate(user)
        )
    except ValueError as e:
        logger.warning(f"Registration validation error for {user_data.email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error for {user_data.email}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    response: Response,
    request: Request,
    db: Session = Depends(get_db)
):
    """Login a user."""
    # Apply rate limiting
    await check_rate_limit(request, settings.RATE_LIMIT_AUTH_PER_MINUTE)
    user = AuthService.authenticate_user(db, credentials)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    # Create session
    session, token = AuthService.create_session(db, user.id)

    # Set cookie with secure settings based on environment
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        secure=settings.COOKIE_SECURE or settings.is_production,
        samesite=settings.COOKIE_SAMESITE,
        max_age=7 * 24 * 60 * 60  # 7 days
    )

    return TokenResponse(message="Login successful")


@router.post("/logout")
async def logout(response: Response, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Logout a user."""
    # The session_token cookie will be passed through get_current_user
    # We need to get it from the request context
    response.delete_cookie("session_token")
    return {"message": "Logout successful"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: User = Depends(get_current_user)):
    """Get current authenticated user information."""
    return UserResponse.model_validate(user)


@router.get("/user")
async def get_user(user: User = Depends(get_current_user)):
    """Get current user (compatibility endpoint)."""
    return {"id": user.id, "email": user.email, "role": user.role.value}


@router.get("/verify-session")
async def verify_session(user: User = Depends(get_current_user)):
    """Verify if the current session is valid."""
    return {
        "valid": True,
        "userId": user.id,
        "userEmail": user.email
    }


@router.post("/password-reset/request")
async def request_password_reset(
    request_data: PasswordResetRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Request a password reset."""
    # Apply rate limiting
    await check_rate_limit(request, settings.RATE_LIMIT_AUTH_PER_MINUTE)
    reset_token = AuthService.create_password_reset_token(db, request_data.email)

    # Always return the same message regardless of whether email exists
    # This prevents email enumeration attacks
    # In production, send email with reset link
    return {
        "message": "If the email exists, a password reset link has been sent"
    }


@router.post("/password-reset/confirm")
async def confirm_password_reset(reset_data: PasswordResetConfirm, db: Session = Depends(get_db)):
    """Confirm password reset with token."""
    success = AuthService.reset_password(db, reset_data.token, reset_data.new_password)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )

    return {"message": "Password reset successful"}

