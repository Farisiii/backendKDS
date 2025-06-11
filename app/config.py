import os
from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Supabase configuration
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # CORS configuration
    cors_origins: list = ["*"]
    cors_credentials: bool = True
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    # Application settings
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_extensions: list = ['.fasta', '.fa', '.fas', '.fna']
    
    class Config:
        env_file = ".env"
    
    def validate_supabase_config(self) -> bool:
        """Validate Supabase configuration"""
        return bool(self.supabase_url and self.supabase_key)

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    
    # Validate required configurations
    if not settings.validate_supabase_config():
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    
    return settings