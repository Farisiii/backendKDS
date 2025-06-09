from fastapi import APIRouter
from datetime import datetime
from app.config import get_settings
from storage.memory_storage import get_storage
from app.services.analysis_service import get_system_statistics

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    settings = get_settings()
    storage = get_storage()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "supabase_configured": settings.validate_supabase_config(),
        "storage_items": storage.get_storage_stats()
    }

@router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    return get_system_statistics()