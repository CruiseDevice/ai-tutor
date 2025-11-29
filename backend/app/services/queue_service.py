"""
Queue service for monitoring and managing ARQ job queue.
"""
from arq import create_pool
from arq.connections import ArqRedis
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..workers.arq_config import ARQ_REDIS_SETTINGS
from ..models.document import Document
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
import logging

logger = logging.getLogger(__name__)


class QueueService:
    """Service for monitoring and managing ARQ job queue."""

    def __init__(self):
        self.redis_pool: Optional[ArqRedis] = None

    async def get_redis_pool(self) -> ArqRedis:
        """Get or create Redis connection pool."""
        if not self.redis_pool:
            self.redis_pool = await create_pool(ARQ_REDIS_SETTINGS)
        return self.redis_pool

    async def get_queue_stats(self, db: Session) -> Dict[str, Any]:
        """
        Get comprehensive queue statistics.

        Returns:
            {
                "queue_depth": int,
                "jobs_processing": int,
                "jobs_pending": int,
                "jobs_completed_1h": int,
                "jobs_failed_1h": int,
                "worker_count": int,
                "workers_active": int,
                "avg_processing_time": float (seconds),
                "success_rate_24h": float (percentage),
                "queue_health": str (healthy/degraded/unhealthy),
                "timestamp": str (ISO format)
            }
        """
        redis = await self.get_redis_pool()

        # Get queue length (pending jobs)
        queue_length = await redis.llen('arq:queue')

        # Get in-progress jobs
        in_progress_keys = await redis.keys('arq:in-progress:*')
        jobs_processing = len(in_progress_keys)

        # Query database for completed/failed jobs in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)

        completed_count = db.query(func.count(Document.id)).filter(
            and_(
                Document.status == 'completed',
                Document.updated_at >= one_hour_ago
            )
        ).scalar()

        failed_count = db.query(func.count(Document.id)).filter(
            and_(
                Document.status == 'failed',
                Document.updated_at >= one_hour_ago
            )
        ).scalar()

        # Calculate average processing time from recent jobs
        avg_time = await self._calculate_avg_processing_time(db)

        # Calculate success rate (last 24 hours)
        success_rate = await self._calculate_success_rate(db)

        # Determine queue health
        queue_health = self._assess_queue_health(
            queue_length,
            jobs_processing,
            failed_count,
            success_rate
        )

        # Worker count (from ARQ settings)
        worker_count = 4  # From arq_config.py max_jobs
        workers_active = min(jobs_processing, worker_count)

        return {
            "queue_depth": queue_length,
            "jobs_processing": jobs_processing,
            "jobs_pending": queue_length,
            "jobs_completed_1h": completed_count,
            "jobs_failed_1h": failed_count,
            "worker_count": worker_count,
            "workers_active": workers_active,
            "avg_processing_time": avg_time,
            "success_rate_24h": success_rate,
            "queue_health": queue_health,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific job.

        Args:
            job_id: ARQ job ID

        Returns:
            Job details dict or None if not found
        """
        redis = await self.get_redis_pool()

        # Get job info from ARQ
        job_info = await redis.get(f'arq:job:{job_id}')

        if not job_info:
            return None

        return {
            "job_id": job_id,
            "status": job_info.get("status"),
            "enqueue_time": job_info.get("enqueue_time"),
            "start_time": job_info.get("start_time"),
            "finish_time": job_info.get("finish_time"),
            "function": job_info.get("function"),
            "args": job_info.get("args"),
            "result": job_info.get("result"),
            "error": job_info.get("error")
        }

    async def get_all_jobs(
        self,
        status_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get list of all jobs with optional status filter.

        Args:
            status_filter: Filter by status (queued, in_progress, complete, failed)
            limit: Maximum number of jobs to return

        Returns:
            List of job dictionaries
        """
        redis = await self.get_redis_pool()
        jobs = []

        if status_filter in ['queued', None]:
            # Get queued jobs
            queued_job_ids = await redis.lrange('arq:queue', 0, limit)
            for job_id in queued_job_ids:
                job_details = await self.get_job_details(job_id.decode() if isinstance(job_id, bytes) else job_id)
                if job_details:
                    jobs.append(job_details)

        if status_filter in ['in_progress', None]:
            # Get in-progress jobs
            in_progress_keys = await redis.keys('arq:in-progress:*')
            for key in in_progress_keys[:limit]:
                job_id = key.decode().split(':')[-1] if isinstance(key, bytes) else key.split(':')[-1]
                job_details = await self.get_job_details(job_id)
                if job_details:
                    jobs.append(job_details)

        return jobs[:limit]

    async def retry_job(self, document_id: str, db: Session) -> Dict[str, Any]:
        """
        Retry a failed job.

        Args:
            document_id: Document ID to retry processing
            db: Database session

        Returns:
            New job info
        """
        redis = await self.get_redis_pool()

        # Get document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise ValueError(f"Document {document_id} not found")

        # Queue new job
        job = await redis.enqueue_job(
            "process_document_job",
            document_id
        )

        # Update document status
        document.job_id = job.job_id
        document.status = "queued"
        document.error_message = None
        db.commit()

        return {
            "document_id": document_id,
            "job_id": job.job_id,
            "status": "queued",
            "message": "Job requeued successfully"
        }

    async def cancel_job(self, job_id: str, db: Session) -> Dict[str, Any]:
        """
        Cancel a job (if still in queue or processing).

        Args:
            job_id: ARQ job ID to cancel
            db: Database session

        Returns:
            Cancellation status
        """
        redis = await self.get_redis_pool()

        # Find document with this job_id
        document = db.query(Document).filter(Document.job_id == job_id).first()

        # Remove from queue if pending
        removed = await redis.lrem('arq:queue', 0, job_id)

        if removed:
            if document:
                document.status = "failed"
                document.error_message = "Cancelled by admin"
                db.commit()

            return {
                "job_id": job_id,
                "status": "cancelled",
                "message": "Job removed from queue"
            }

        # If job is in-progress, we can't truly cancel it
        # but we can mark the document as failed
        in_progress = await redis.get(f'arq:in-progress:{job_id}')

        if in_progress and document:
            document.status = "failed"
            document.error_message = "Cancelled by admin (was in progress)"
            db.commit()

            return {
                "job_id": job_id,
                "status": "marked_cancelled",
                "message": "Job was processing, marked as failed"
            }

        return {
            "job_id": job_id,
            "status": "not_found",
            "message": "Job not found in queue"
        }

    async def _calculate_avg_processing_time(self, db: Session) -> float:
        """
        Calculate average processing time for completed jobs (last 24h).

        Uses processing_started_at and processing_completed_at columns
        to calculate real processing durations.

        Returns:
            Average processing time in seconds, or 120.0 if no data available
        """
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)

        # Query completed documents that have processing time data
        documents = db.query(Document).filter(
            and_(
                Document.status == 'completed',
                Document.processing_started_at.isnot(None),
                Document.processing_completed_at.isnot(None),
                Document.updated_at >= twenty_four_hours_ago
            )
        ).all()

        if not documents:
            # No data available - return default estimate
            return 120.0

        # Calculate average duration in seconds
        total_duration = 0.0
        for doc in documents:
            if doc.processing_duration:
                total_duration += doc.processing_duration

        avg_duration = total_duration / len(documents)
        return round(avg_duration, 2)

    async def _calculate_success_rate(self, db: Session) -> float:
        """Calculate success rate over last 24 hours."""
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)

        total = db.query(func.count(Document.id)).filter(
            and_(
                Document.status.in_(['completed', 'failed']),
                Document.updated_at >= twenty_four_hours_ago
            )
        ).scalar()

        if total == 0:
            return 100.0

        completed = db.query(func.count(Document.id)).filter(
            and_(
                Document.status == 'completed',
                Document.updated_at >= twenty_four_hours_ago
            )
        ).scalar()

        return (completed / total) * 100.0

    def _assess_queue_health(
        self,
        queue_depth: int,
        processing: int,
        failed_1h: int,
        success_rate: float
    ) -> str:
        """
        Assess overall queue health.

        Returns: 'healthy', 'degraded', or 'unhealthy'
        """
        # Unhealthy conditions
        if queue_depth > 50:  # Too many pending jobs
            return "unhealthy"
        if success_rate < 70:  # Low success rate
            return "unhealthy"
        if failed_1h > 10:  # Too many recent failures
            return "unhealthy"

        # Degraded conditions
        if queue_depth > 20:
            return "degraded"
        if success_rate < 90:
            return "degraded"
        if failed_1h > 3:
            return "degraded"

        return "healthy"


# Singleton instance
queue_service = QueueService()
