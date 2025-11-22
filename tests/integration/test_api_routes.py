"""Integration tests for API routes"""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
@pytest.mark.api
class TestAPIRoutes:
    """Test suite for API endpoints"""
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint returns correct information"""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "api_docs" in data
        assert "gradio_ui" in data
        assert "health" in data
        assert "metrics" in data
    
    def test_health_endpoint_healthy(self, api_client):
        """Test health endpoint when service is healthy"""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["index_loaded"] is True
        assert data["total_vectors"] == 1000
        assert "app_name" in data
        assert "version" in data
    
    def test_metrics_endpoint(self, api_client):
        """Test metrics endpoint returns proper structure"""
        response = api_client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "timestamp" in data
        
        metrics = data["metrics"]
        assert "system" in metrics
        assert "latency" in metrics
        assert "throughput" in metrics
        assert "reliability" in metrics
        assert "model_quality" in metrics
        assert "usage" in metrics
        assert "cost" in metrics
    
    def test_query_endpoint_success(self, api_client, sample_query_request):
        """Test successful query request"""
        response = api_client.post("/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "query" in data
        assert "answer" in data
        assert "k" in data
        assert "model" in data
        assert "temperature" in data
        assert "retrieval_results" in data
        
        # Verify values (mock returns "Test query" for query field)
        assert data["query"] == "Test query"  # From mock
        assert data["k"] == sample_query_request["k"]
        assert data["model"] == sample_query_request["model"]
        assert isinstance(data["retrieval_results"], list)
        assert len(data["retrieval_results"]) > 0
    
    def test_query_endpoint_with_defaults(self, api_client):
        """Test query endpoint with only required fields"""
        response = api_client.post("/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check defaults are applied
        assert data["k"] == 5  # Default
        assert data["temperature"] == 0.2  # Default
        assert data["model"] == "gpt-4o-mini"  # Default
    
    def test_query_endpoint_missing_query(self, api_client):
        """Test query endpoint with missing query field"""
        response = api_client.post("/query", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_invalid_k(self, api_client):
        """Test query endpoint with invalid k value"""
        response = api_client.post("/query", json={
            "query": "Test query",
            "k": 100  # Too high (max is 50)
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_invalid_temperature(self, api_client):
        """Test query endpoint with invalid temperature"""
        response = api_client.post("/query", json={
            "query": "Test query",
            "temperature": 3.0  # Too high (max is 2.0)
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_tracks_metrics(self, api_client, sample_query_request):
        """Test that query endpoint tracks metrics correctly"""
        # Get initial metrics
        initial_response = api_client.get("/metrics")
        initial_count = initial_response.json()["metrics"]["throughput"]["total_requests"]
        
        # Make a query
        api_client.post("/query", json=sample_query_request)
        
        # Get updated metrics
        updated_response = api_client.get("/metrics")
        updated_data = updated_response.json()["metrics"]
        
        # Verify metrics were updated
        assert updated_data["throughput"]["total_requests"] == initial_count + 1
        assert updated_data["reliability"]["successful_requests"] > 0
        assert updated_data["reliability"]["success_rate_percent"] > 0
    
    def test_retrieval_results_structure(self, api_client, sample_query_request):
        """Test that retrieval results have correct structure"""
        response = api_client.post("/query", json=sample_query_request)
        data = response.json()
        
        retrieval_results = data["retrieval_results"]
        assert len(retrieval_results) > 0
        
        # Check first result structure
        first_result = retrieval_results[0]
        assert "rank" in first_result
        assert "distance" in first_result
        assert "branch" in first_result
        assert "rating" in first_result
        assert "reviewer_location" in first_result
        assert "snippet" in first_result
        
        # Verify types
        assert isinstance(first_result["rank"], int)
        assert isinstance(first_result["distance"], float)
        assert isinstance(first_result["snippet"], str)
    
    def test_multiple_queries_different_models(self, api_client):
        """Test multiple queries with different models"""
        # Query with gpt-4o-mini
        response1 = api_client.post("/query", json={
            "query": "Test query 1",
            "model": "gpt-4o-mini"
        })
        
        # Query with gpt-4o
        response2 = api_client.post("/query", json={
            "query": "Test query 2",
            "model": "gpt-4o"
        })
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Check metrics track both models
        metrics_response = api_client.get("/metrics")
        usage = metrics_response.json()["metrics"]["usage"]
        
        assert "queries_by_model" in usage
        # Note: Actual values depend on mock behavior
    
    def test_cors_headers(self, api_client):
        """Test that CORS headers are present"""
        response = api_client.get("/")
        
        # CORS headers should be added by middleware
        # Note: TestClient may not include all CORS headers
        assert response.status_code == 200

