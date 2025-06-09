from fastapi import FastAPI
from app.config import get_settings
from app.dependencies import setup_middleware
from app.routers import health, upload, data, analysis
import warnings

warnings.filterwarnings('ignore')

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="GenMAP",
        version="1.0.0",
        description="Genetic Mapping and Population Analysis Platform"
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(upload.router, prefix="/upload", tags=["upload"])
    app.include_router(data.router, prefix="/data", tags=["data"])
    app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
    
    @app.get("/")
    async def root():
        return {
            "message": "GenMAP API",
            "version": "1.0.0",
            "status": "running"
        }
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)