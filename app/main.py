"""Main FastAPI application - mounts Gradio UI at /ui By Yigal"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from app.core.config import settings
from app.services.rag_builder import RAGBuilder
from app.services.rag_query import RAGQueryHandler
from app.api.routes import router, set_query_handler as set_api_query_handler
from app.ui.gradio_interface import create_gradio_interface, set_query_handler as set_ui_query_handler
from app.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(
    log_level=getattr(logging, settings.log_level.upper()),
    log_to_file=settings.log_to_file,
    log_dir=str(settings.log_dir)
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown"""
    # ============================================================
    # STARTUP
    # ============================================================
    logger.info("=" * 80)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Build or load RAG index
        logger.info("Step 1: Building/Loading RAG index...")
        builder = RAGBuilder(settings)
        index, metadata = builder.build_or_load()
        logger.info(f"✓ Index ready: {index.ntotal} vectors")
        
        # Step 2: Initialize query handler
        logger.info("Step 2: Initializing query handler...")
        query_handler = RAGQueryHandler(settings, index, metadata)
        logger.info("✓ Query handler ready")
        
        # Step 3: Set query handler for both API and UI
        logger.info("Step 3: Connecting services...")
        set_api_query_handler(query_handler)  # For REST API
        set_ui_query_handler(query_handler)   # For Gradio UI
        logger.info("✓ Services connected")
        
        logger.info("=" * 80)
        logger.info("🚀 Application ready!")
        logger.info(f"📖 API Documentation: http://localhost:8000/docs")
        logger.info(f"🎨 Gradio UI: http://localhost:8000/ui")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # ============================================================
    # SHUTDOWN
    # ============================================================
    logger.info("=" * 80)
    logger.info(f"Shutting down {settings.app_name}")
    logger.info("=" * 80)


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG API for querying Disneyland reviews with integrated Gradio UI",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API routes
app.include_router(router)

# Mount Gradio UI at /ui
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

logger.info("Gradio UI mounted at /ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
