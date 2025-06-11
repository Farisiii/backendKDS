from fastapi import APIRouter, HTTPException, UploadFile, File
from datetime import datetime
from app.config import get_settings
from app.utils.parsers import parse_fasta_content, create_demo_fasta
from storage.memory_storage import get_storage

router = APIRouter()

@router.post("/demo")
async def upload_demo_data():
    """Upload demo data for testing"""
    storage = get_storage()
    
    demo_fasta = create_demo_fasta()
    data_id = "demo_data"
    uploaded_data = parse_fasta_content(demo_fasta, "demo_species.fasta")
    
    storage.store_uploaded_data(data_id, uploaded_data)
    
    return {"dataId": data_id, "data": uploaded_data}

@router.post("/")
async def upload_fasta_file(file: UploadFile = File(...)):
    """Upload FASTA file with validation"""
    settings = get_settings()
    storage = get_storage()
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_file_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"File must be in FASTA format {settings.allowed_file_extensions}"
        )
    
    try:
        content = await file.read()
        
        # Check file size
        if len(content) > settings.max_upload_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.max_upload_size // (1024*1024)}MB"
            )
        
        content_str = content.decode('utf-8')
        
        # Basic validation
        if not content_str.strip():
            raise HTTPException(status_code=400, detail="File is empty")
        
        if not content_str.strip().startswith('>'):
            raise HTTPException(status_code=400, detail="Invalid FASTA format - must start with '>'")
        
        # Generate unique data ID
        data_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        uploaded_data = parse_fasta_content(content_str, file.filename)
        uploaded_data.isDemo = False
        
        # Store data
        storage.store_uploaded_data(data_id, uploaded_data)
        
        if len(uploaded_data.sequences) == 0:
            raise HTTPException(status_code=400, detail="No valid sequences found in file")
        
        return {"dataId": data_id, "data": uploaded_data}
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported - please use UTF-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")