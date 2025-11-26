from fastapi import APIRouter
from ..config import settings

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("")
async def get_config():
    """Get public configuration values for the frontend."""
    return {
        "max_file_size": settings.MAX_FILE_SIZE,
        "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024)
    }

