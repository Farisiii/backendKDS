from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from app.config import get_settings

def setup_middleware(app: FastAPI) -> None:
    """Setup middleware for the FastAPI application"""
    settings = get_settings()
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)