from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import uuid
from ..models.user import User, Session as DBSession, PasswordResetToken
from ..core.security import verify_password, get_password_hash, create_access_token
from ..schemas.auth import UserCreate, UserLogin


class AuthService:
    @staticmethod
    def create_user(db: Session, user_data: UserCreate) -> User:
        """Create a new user."""
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user
        user = User(
            email=user_data.email,
            password=hashed_password
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return user
    
    @staticmethod
    def authenticate_user(db: Session, credentials: UserLogin) -> Optional[User]:
        """Authenticate a user with email and password."""
        user = db.query(User).filter(User.email == credentials.email).first()
        
        if not user:
            return None
        
        if not verify_password(credentials.password, user.password):
            return None
        
        return user
    
    @staticmethod
    def create_session(db: Session, user_id: str) -> tuple[DBSession, str]:
        """Create a new session for a user and return session and JWT token."""
        # Generate JWT token
        token = create_access_token(data={"sub": user_id})
        
        # Create session in database
        session = DBSession(
            user_id=user_id,
            token=token,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return session, token
    
    @staticmethod
    def delete_session(db: Session, token: str) -> bool:
        """Delete a session."""
        session = db.query(DBSession).filter(DBSession.token == token).first()
        if session:
            db.delete(session)
            db.commit()
            return True
        return False
    
    @staticmethod
    def verify_session(db: Session, token: str) -> Optional[User]:
        """Verify a session token and return the user."""
        session = db.query(DBSession).filter(DBSession.token == token).first()
        
        if not session or session.expires_at < datetime.utcnow():
            return None
        
        user = db.query(User).filter(User.id == session.user_id).first()
        return user
    
    @staticmethod
    def create_password_reset_token(db: Session, email: str) -> Optional[PasswordResetToken]:
        """Create a password reset token for a user."""
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return None
        
        # Generate token
        token_str = str(uuid.uuid4())
        
        # Create reset token
        reset_token = PasswordResetToken(
            email=email,
            token=token_str,
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        db.add(reset_token)
        db.commit()
        db.refresh(reset_token)
        
        return reset_token
    
    @staticmethod
    def reset_password(db: Session, token: str, new_password: str) -> bool:
        """Reset a user's password using a reset token."""
        reset_token = db.query(PasswordResetToken).filter(
            PasswordResetToken.token == token,
            PasswordResetToken.used == False
        ).first()
        
        if not reset_token or reset_token.expires_at < datetime.utcnow():
            return False
        
        # Get user
        user = db.query(User).filter(User.email == reset_token.email).first()
        if not user:
            return False
        
        # Update password
        user.password = get_password_hash(new_password)
        reset_token.used = True
        
        db.commit()
        
        return True

