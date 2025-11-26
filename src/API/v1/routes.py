from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional
import json

from .endpoints import auth, documents, users, analytics

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])

@api_router.get("/status", tags=["System"])
async def api_status():
    """API status endpoint"""
    return {
        "status": "operational",
        "version": "7.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }
