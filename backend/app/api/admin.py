"""
Admin API endpoints for queue monitoring and management.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
import asyncio
import json
import logging

from ..database import get_db
from ..core.deps import get_admin_user, get_super_admin_user
from ..models.user import User
from ..services.queue_service import queue_service
from ..services.audit_service import log_admin_action

router = APIRouter(prefix="/api/admin", tags=["admin"])
logger = logging.getLogger(__name__)


@router.get("/queue/stats")
async def get_queue_statistics(
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive queue statistics.
    Requires admin role.
    """
    try:
        stats = await queue_service.get_queue_stats(db)
        return stats
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve queue statistics: {str(e)}"
        )


@router.get("/queue/jobs")
async def list_jobs(
    status_filter: Optional[str] = None,
    limit: int = 100,
    admin: User = Depends(get_admin_user)
):
    """
    List all jobs with optional status filter.

    Query params:
        - status_filter: queued, in_progress, complete, failed
        - limit: max number of jobs to return (default 100)
    """
    try:
        jobs = await queue_service.get_all_jobs(status_filter, limit)
        return {
            "jobs": jobs,
            "count": len(jobs),
            "filter": status_filter
        }
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.get("/queue/jobs/{job_id}")
async def get_job_details(
    job_id: str,
    admin: User = Depends(get_admin_user)
):
    """Get detailed information about a specific job."""
    try:
        job = await queue_service.get_job_details(job_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )

        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job details: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job details: {str(e)}"
        )


@router.post("/queue/jobs/{document_id}/retry")
async def retry_failed_job(
    document_id: str,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Retry processing a failed document.
    Creates a new job in the queue.
    """
    try:
        result = await queue_service.retry_job(document_id, db)

        # Log the admin action
        log_admin_action(
            db,
            user_id=admin.id,
            action="retry_job",
            resource_type="document",
            resource_id=document_id,
            details=f"Requeued job {result.get('job_id', 'unknown')}"
        )

        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to retry job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry job: {str(e)}"
        )


@router.delete("/queue/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    admin: User = Depends(get_super_admin_user),
    db: Session = Depends(get_db)
):
    """
    Cancel a job (remove from queue or mark as failed).
    Requires super admin role.
    """
    try:
        result = await queue_service.cancel_job(job_id, db)

        # Log the admin action
        log_admin_action(
            db,
            user_id=admin.id,
            action="cancel_job",
            resource_type="job",
            resource_id=job_id,
            details=f"Status: {result.get('status', 'unknown')}, Message: {result.get('message', 'N/A')}"
        )

        return result
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )


@router.get("/queue/stream")
async def stream_queue_stats(
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Stream real-time queue statistics via Server-Sent Events.
    Updates every 2 seconds.
    """
    async def event_generator():
        try:
            while True:
                # Get current stats
                stats = await queue_service.get_queue_stats(db)

                # Send as SSE event
                yield f"data: {json.dumps(stats)}\n\n"

                # Wait before next update
                await asyncio.sleep(2)

        except asyncio.CancelledError:
            logger.info("SSE connection closed by client")
        except Exception as e:
            logger.error(f"Error in SSE stream: {e}", exc_info=True)
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
