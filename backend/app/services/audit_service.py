"""
Audit logging service for tracking admin actions.

This service provides centralized audit logging for administrative
actions, compliance, and security monitoring.
"""
import logging
from typing import Optional
from sqlalchemy.orm import Session

from ..models.audit_log import AuditLog

logger = logging.getLogger(__name__)


def log_admin_action(
    db: Session,
    user_id: str,
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[str] = None
) -> AuditLog:
    """
    Log an administrative action to the audit trail.

    Args:
        db: Database session
        user_id: ID of the user performing the action
        action: Action being performed (e.g., "retry_job", "cancel_job", "view_stats")
        resource_type: Type of resource (e.g., "document", "job", "user")
        resource_id: ID of the resource being acted upon
        details: Additional details (e.g., JSON string with extra context)

    Returns:
        Created AuditLog entry

    Examples:
        >>> log_admin_action(
        ...     db,
        ...     user_id="admin-123",
        ...     action="retry_job",
        ...     resource_type="document",
        ...     resource_id="doc-456",
        ...     details="Requeued failed document processing job"
        ... )
    """
    try:
        audit_entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details
        )

        db.add(audit_entry)
        db.commit()

        logger.info(
            f"AUDIT: User {user_id} performed '{action}' on "
            f"{resource_type or 'unknown'} {resource_id or 'N/A'}"
        )

        return audit_entry

    except Exception as e:
        logger.error(f"Failed to log audit entry: {e}", exc_info=True)
        db.rollback()
        # Don't raise - audit logging failures shouldn't break the main operation
        # But we do log the error for debugging
        return None


def get_audit_logs(
    db: Session,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> list[AuditLog]:
    """
    Retrieve audit logs with optional filtering.

    Args:
        db: Database session
        user_id: Filter by user ID
        action: Filter by action type
        resource_type: Filter by resource type
        limit: Maximum number of logs to return
        offset: Number of logs to skip

    Returns:
        List of AuditLog entries
    """
    query = db.query(AuditLog)

    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if action:
        query = query.filter(AuditLog.action == action)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)

    # Order by most recent first
    query = query.order_by(AuditLog.timestamp.desc())

    # Apply pagination
    query = query.offset(offset).limit(limit)

    return query.all()


def get_audit_log_count(
    db: Session,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    resource_type: Optional[str] = None
) -> int:
    """
    Count audit logs matching filters.

    Args:
        db: Database session
        user_id: Filter by user ID
        action: Filter by action type
        resource_type: Filter by resource type

    Returns:
        Count of matching audit logs
    """
    query = db.query(AuditLog)

    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if action:
        query = query.filter(AuditLog.action == action)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)

    return query.count()
