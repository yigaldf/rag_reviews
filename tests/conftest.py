"""Pytest configuration and shared fixtures"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock
import numpy as np


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    mock = Mock()
    mock.app_name = "Test RAG API"
    mock.app_version = "1.0.0-test"
    mock.data_file = "test_data.csv"
    mock.index_dir = "test_index"
    mock.chunk_size = 500
    mock.chunk_overlap = 50
    mock.embedding_model = "text-embedding-3-small"
    mock.openai_api_key = "test-key"
    mock.top_k = 5
    mock.log_level = "INFO"
    mock.log_to_file = False
    mock.log_dir = "test_logs"
    mock.debug = True
    return mock


@pytest.fixture
def mock_query_handler():
    """Mock RAG query handler"""
    handler = MagicMock()
    handler.total_vectors = 1000
    handler.answer_query.return_value = {
        "query": "Test query",
        "answer": "This is a test answer about Disneyland.",
        "k": 5,
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "retrieval_results": [
            {
                "rank": 1,
                "distance": 0.234,
                "branch": "Disneyland_HongKong",
                "rating": 5.0,
                "reviewer_location": "Singapore",
                "snippet": "Amazing park with great attractions..."
            },
            {
                "rank": 2,
                "distance": 0.298,
                "branch": "Disneyland_California",
                "rating": 4.0,
                "reviewer_location": "USA",
                "snippet": "Good experience overall, some long wait times..."
            }
        ]
    }
    return handler


@pytest.fixture
def mock_faiss_index():
    """Mock FAISS index"""
    index = MagicMock()
    index.ntotal = 1000
    # Mock search results: distances and indices
    index.search.return_value = (
        np.array([[0.234, 0.298, 0.345, 0.412, 0.456]]),  # distances
        np.array([[10, 25, 42, 88, 120]])  # indices
    )
    return index


@pytest.fixture
def mock_metadata():
    """Mock metadata for reviews"""
    return [
        {
            "Review_ID": i,
            "Rating": 5.0 if i % 2 == 0 else 4.0,
            "Year_Month": "2023-01",
            "Reviewer_Location": "Singapore" if i % 2 == 0 else "USA",
            "Branch": "Disneyland_HongKong" if i % 3 == 0 else "Disneyland_California",
            "chunk": f"This is test review chunk {i}. It talks about various aspects of the park."
        }
        for i in range(150)
    ]


@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What do visitors like about Disneyland Hong Kong?",
        "k": 5,
        "temperature": 0.2,
        "model": "gpt-4o-mini"
    }


@pytest.fixture
def api_client(mock_query_handler):
    """Test client for API with mocked query handler"""
    from app.api.routes import router, set_query_handler
    from fastapi import FastAPI
    
    # Create a test app
    app = FastAPI()
    app.include_router(router)
    
    # Set mock query handler
    set_query_handler(mock_query_handler)
    
    # Return test client
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before each test"""
    from app.utils.metrics import MetricsCollector
    from app.api import routes
    
    # Reset the global metrics instance
    routes.metrics = MetricsCollector()
    
    yield
    
    # Clean up after test
    routes.metrics = MetricsCollector()

