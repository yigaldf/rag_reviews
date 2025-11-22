"""REST API routes"""
import logging
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import QueryRequest, QueryResponse, HealthResponse
from app.utils.metrics import MetricsCollector

logger = logging.getLogger('rag_system.api')

router = APIRouter()

# Global query handler (set at startup)
query_handler = None

# Monitoring metrics
metrics = MetricsCollector()


def set_query_handler(handler):
    """Set the global query handler"""
    global query_handler
    query_handler = handler
    logger.info("Query handler set for API routes")


@router.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information and available routes.
    
    The Gradio UI is available at `/ui` (not shown in Swagger as it's a mounted sub-app).
    """
    return {
        "message": "Disneyland Reviews RAG API",
        "api_docs": "/docs",
        "gradio_ui": "/ui",
        "health": "/health",
        "metrics": "/metrics"
    }


@router.get("/ui-info", tags=["UI"], include_in_schema=True)
async def ui_info():
    """
    Information about the Gradio UI.
    
    The interactive Gradio interface is mounted at `/ui` as a separate ASGI application.
    It provides a user-friendly web interface for querying the RAG system.
    
    To access it, navigate to: http://localhost:8000/ui
    
    Features:
    - Interactive query input
    - Real-time results
    - Adjustable parameters (k, temperature, model)
    - Visual display of retrieval results
    """
    return {
        "ui_path": "/ui",
        "type": "Gradio Interface",
        "description": "Interactive web UI for RAG queries",
        "features": [
            "Real-time query processing",
            "Adjustable retrieval parameters",
            "Visual result display",
            "No API key needed"
        ],
        "note": "The UI is mounted as a separate ASGI app and doesn't appear in OpenAPI schema"
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if query_handler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized"
        )
    
    return HealthResponse(
        status="healthy",
        app_name="Disneyland Reviews RAG API",
        version="1.0.0",
        index_loaded=True,
        total_vectors=query_handler.total_vectors
    )


@router.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get API metrics and statistics.
    
    Returns performance metrics including:
    - Request counts (total, successful, failed)
    - Response time statistics (average, min, max, percentiles)
    - Usage patterns (by model, by k value)
    - Uptime information
    """
    return {
        "metrics": metrics.get_stats(),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_reviews(request: QueryRequest):
    """
    Query Disneyland reviews using RAG.
    
    Returns an answer based on relevant review chunks retrieved from the FAISS index.
    """
    start_time = time.time()
    success = False
    error_type = None
    result = None
    
    if query_handler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized"
        )
    
    try:
        logger.info(f"API Query: {request.query[:100]}...")
        
        result = query_handler.answer_query(
            query=request.query,
            k=request.k,
            temperature=request.temperature,
            model=request.model
        )
        
        response = QueryResponse(**result)
        logger.info("Query processed successfully")
        
        success = True
        return response
        
    except HTTPException as e:
        error_type = f"http_{e.status_code}"
        raise
    except TimeoutError:
        error_type = "timeout"
        logger.error(f"Timeout processing query", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Query processing timeout"
        )
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )
    finally:
        # Calculate comprehensive metrics
        response_time = time.time() - start_time
        query_length = len(request.query)
        
        # Extract additional metrics from result if available
        avg_distance = None
        response_length = None
        
        if result and success:
            # Calculate average retrieval distance
            if 'retrieval_results' in result and result['retrieval_results']:
                distances = [r['distance'] for r in result['retrieval_results']]
                avg_distance = sum(distances) / len(distances)
            
            # Get response length
            if 'answer' in result:
                response_length = len(result['answer'])
        
        # Record comprehensive metrics
        metrics.record_request(
            success=success,
            response_time=response_time,
            model=request.model,
            k=request.k,
            error_type=error_type,
            avg_retrieval_distance=avg_distance,
            query_length=query_length,
            response_length=response_length,
            tokens_used=None  # Could extract from OpenAI response if available
        )
        
        # Format distance for logging
        distance_str = f"{avg_distance:.4f}" if avg_distance is not None else "N/A"
        logger.info(
            f"Request completed in {response_time:.3f}s "
            f"(success={success}, avg_distance={distance_str})"
        )
