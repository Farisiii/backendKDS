from supabase import create_client, Client
import os
from typing import Optional

# Global Supabase client
supabase: Optional[Client] = None

def initialize_supabase():
    """Initialize Supabase client"""
    global supabase
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    if supabase is None:
        raise ValueError("Supabase client not initialized")
    return supabase

async def upload_to_supabase(file_content: bytes, filename: str, bucket: str = "visualizations") -> str:
    """Upload file to Supabase storage"""
    try:
        client = get_supabase_client()
        result = client.storage.from_(bucket).upload(filename, file_content)
        if result.status_code == 200:
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"
        else:
            raise Exception(f"Upload failed: {result}")
    except Exception as e:
        print(f"Supabase upload error: {e}")
        return ""