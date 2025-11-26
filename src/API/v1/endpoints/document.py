from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from typing import List, Optional, Dict, Any
import json
import uuid
from datetime import datetime

from src.services.ai_engine import EnhancedAIModelManager
from src.services.security_engine import EnhancedSecurityEngine
from src.core.logging import EnhancedProductionLogger
from src.config.settings import settings

router = APIRouter()

# This would typically be dependency injection
async def get_ai_engine():
    from src.main import ai_engine
    return ai_engine

async def get_security_engine():
    from src.main import security_engine
    return security_engine

async def get_logger():
    from src.main import production_logger
    return production_logger

@router.post("/process", response_model=Dict[str, Any])
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    processing_types: str = Form("entities,sentiment,summary"),
    user_id: str = Form(...),
    ai_engine: EnhancedAIModelManager = Depends(get_ai_engine),
    security_engine: EnhancedSecurityEngine = Depends(get_security_engine),
    logger: EnhancedProductionLogger = Depends(get_logger)
):
    """Enhanced document processing endpoint"""
    try:
        # Security validation
        threat_check = await security_engine.detect_threats({
            'ip_address': 'client_ip',  # Would get from request
            'user_agent': 'user_agent',  # Would get from request
            'action': 'document_processing',
            'user_id': user_id
        })
        
        if threat_check['risk_level'] in ['HIGH', 'CRITICAL']:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Request blocked due to security policy"
            )
        
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Read file content
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Parse processing types
        processing_list = [pt.strip() for pt in processing_types.split(',')]
        
        # Create document object
        document = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "content": content.decode('utf-8'),
            "type": file.content_type or "unknown",
            "user_id": user_id,
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        # Process document
        results = await ai_engine.process_document_enhanced(document, processing_list)
        
        # Log successful processing
        await logger.log_business_event(
            event_type="DOCUMENT_PROCESSED_SUCCESS",
            user_id=user_id,
            details={
                "document_id": document["id"],
                "filename": file.filename,
                "processing_types": processing_list,
                "processing_time": results['metadata'].get('processing_time', {})
            },
            severity="INFO",
            business_value=5.0
        )
        
        return {
            "success": True,
            "document_id": document["id"],
            "results": results['results'],
            "enhanced_insights": results['enhanced_insights'],
            "business_intelligence": results['business_intelligence'],
            "processing_metadata": results['metadata']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await logger.log_security_event(
            event_type="DOCUMENT_PROCESSING_ERROR",
            severity="ERROR",
            user_id=user_id,
            details={
                "error": str(e),
                "filename": file.filename if file else "unknown",
                "processing_types": processing_types
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed"
        )

@router.get("/{document_id}", response_model=Dict[str, Any])
async def get_document_results(
    document_id: str,
    user_id: str,
    security_engine: EnhancedSecurityEngine = Depends(get_security_engine),
    logger: EnhancedProductionLogger = Depends(get_logger)
):
    """Get document processing results"""
    try:
        # Security check
        threat_check = await security_engine.detect_threats({
            'action': 'get_document_results',
            'user_id': user_id,
            'document_id': document_id
        })
        
        if threat_check['risk_level'] in ['HIGH', 'CRITICAL']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied due to security policy"
            )
        
        # In production, this would fetch from database
        # For now, return mock data
        mock_results = {
            "document_id": document_id,
            "status": "processed",
            "results_available": True,
            "processed_at": datetime.utcnow().isoformat()
        }
        
        return mock_results
        
    except HTTPException:
        raise
    except Exception as e:
        await logger.log_security_event(
            event_type="DOCUMENT_RETRIEVAL_ERROR",
            severity="ERROR",
            user_id=user_id,
            details={
                "error": str(e),
                "document_id": document_id
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document results"
        )

@router.get("/{document_id}/insights", response_model=Dict[str, Any])
async def get_document_insights(
    document_id: str,
    user_id: str,
    security_engine: EnhancedSecurityEngine = Depends(get_security_engine),
    logger: EnhancedProductionLogger = Depends(get_logger)
):
    """Get enhanced document insights"""
    try:
        # Security and access validation would go here
        
        # Mock insights response
        insights = {
            "document_id": document_id,
            "key_findings": [
                "Strong positive sentiment detected",
                "Multiple business opportunities identified",
                "Low risk assessment score"
            ],
            "recommendations": [
                "Consider expanding mentioned markets",
                "Monitor competitive landscape",
                "Follow up on partnership opportunities"
            ],
            "risk_assessment": {
                "level": "low",
                "score": 25,
                "factors": ["positive_sentiment", "clear_intent", "professional_tone"]
            }
        }
        
        return insights
        
    except Exception as e:
        await logger.log_security_event(
            event_type="INSIGHTS_RETRIEVAL_ERROR",
            severity="ERROR",
            user_id=user_id,
            details={
                "error": str(e),
                "document_id": document_id
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document insights"
        )
