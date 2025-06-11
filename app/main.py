from fastapi import FastAPI
from config import get_settings
from dependencies import setup_middleware
from routers import health, upload, data, analysis
import warnings

warnings.filterwarnings('ignore')

def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title="GenMAP",
        version="1.0.0",
        description="Genetic Mapping and Population Analysis Platform"
    )
    
    setup_middleware(app)
    
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
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)