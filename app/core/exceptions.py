from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Union
import logging

logger = logging.getLogger(__name__)

class GenMAPException(Exception):
    """Base exception for GenMAP application"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class FileValidationError(GenMAPException):
    """Exception raised for file validation errors"""
    def __init__(self, message: str):
        super().__init__(message, 400)

class AnalysisError(GenMAPException):
    """Exception raised for analysis errors"""
    def __init__(self, message: str):
        super().__init__(message, 500)

class DataNotFoundError(GenMAPException):
    """Exception raised when data is not found"""
    def __init__(self, message: str = "Data not found"):
        super().__init__(message, 404)

class AnalysisNotFoundError(GenMAPException):
    """Exception raised when analysis is not found"""
    def __init__(self, message: str = "Analysis not found"):
        super().__init__(message, 404)

class InsufficientDataError(GenMAPException):
    """Exception raised when there's insufficient data for analysis"""
    def __init__(self, message: str = "Insufficient data for analysis"):
        super().__init__(message, 400)

async def genmap_exception_handler(request: Request, exc: GenMAPException):
    """Handle GenMAP custom exceptions"""
    logger.error(f"GenMAP Exception: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "type": exc.__class__.__name__}
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def setup_exception_handlers(app: FastAPI):
    """Setup exception handlers for the FastAPI app"""
    app.add_exception_handler(GenMAPException, genmap_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)