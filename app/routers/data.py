from fastapi import APIRouter, HTTPException
from storage.memory_storage import get_storage

router = APIRouter()

@router.get("/{data_id}")
async def get_uploaded_data(data_id: str):
    """Get uploaded data by ID"""
    storage = get_storage()
    
    data = storage.get_uploaded_data(data_id)
    if not data:
        raise HTTPException(status_code=404, detail="Data not found")
    
    return data

@router.delete("/{data_id}")
async def delete_uploaded_data(data_id: str):
    """Delete uploaded data and associated analyses"""
    storage = get_storage()
    
    # Check if data exists
    if not storage.get_uploaded_data(data_id):
        raise HTTPException(status_code=404, detail="Data not found")
    
    # Remove uploaded data
    storage.delete_uploaded_data(data_id)
    
    # Remove associated analyses
    deleted_analyses = storage.delete_analyses_by_data_id(data_id)
    
    return {
        "message": f"Data {data_id} deleted successfully",
        "deleted_analyses": deleted_analyses
    }

@router.get("/")
async def list_uploaded_data():
    """List all uploaded data"""
    storage = get_storage()
    data_ids = storage.list_uploaded_data()
    
    result = []
    for data_id in data_ids:
        data = storage.get_uploaded_data(data_id)
        if data:
            result.append({
                "dataId": data_id,
                "filename": data.filename,
                "sequenceCount": len(data.sequences),
                "isDemo": data.isDemo,
                "uploadedAt": data.uploadedAt if hasattr(data, 'uploadedAt') else None
            })
    
    return result