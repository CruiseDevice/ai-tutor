from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.sql import func
from ..database import Base
from datetime import datetime, timezone
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class AuditLog(Base):
    """
    Audit log model for tracking administrative actions.

    Stores a complete audit trail of admin actions for:
    - Compliance and security auditing
    - Debugging and troubleshooting
    - User activity monitoring
    """
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False, index=True)
    resource_type = Column(String, nullable=True)  # e.g., "document", "job", "user"
    resource_id = Column(String, nullable=True)    # ID of the resource being acted upon
    details = Column(Text, nullable=True)          # Additional context (JSON string or text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
