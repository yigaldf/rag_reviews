"""Unit tests for Pydantic schemas"""
import pytest
from pydantic import ValidationError
from app.models.schemas import QueryRequest, QueryResponse, RetrievalResult, HealthResponse


@pytest.mark.unit
class TestQueryRequest:
    """Test suite for QueryRequest schema"""
    
    def test_valid_query_request(self):
        """Test creating a valid query request"""
        request = QueryRequest(
            query="What do visitors like?",
            k=10,
            temperature=0.5,
            model="gpt-4o"
        )
        
        assert request.query == "What do visitors like?"
        assert request.k == 10
        assert request.temperature == 0.5
        assert request.model == "gpt-4o"
    
    def test_query_request_with_defaults(self):
        """Test query request uses default values"""
        request = QueryRequest(query="Test query")
        
        assert request.k == 5
        assert request.temperature == 0.2
        assert request.model == "gpt-4o-mini"
    
    def test_query_request_empty_query(self):
        """Test that empty query is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="")
        
        errors = exc_info.value.errors()
        assert any("min_length" in str(error) for error in errors)
    
    def test_query_request_invalid_k_too_low(self):
        """Test that k < 1 is rejected"""
        with pytest.raises(ValidationError):
            QueryRequest(query="Test", k=0)
    
    def test_query_request_invalid_k_too_high(self):
        """Test that k > 50 is rejected"""
        with pytest.raises(ValidationError):
            QueryRequest(query="Test", k=100)
    
    def test_query_request_invalid_temperature_too_low(self):
        """Test that temperature < 0 is rejected"""
        with pytest.raises(ValidationError):
            QueryRequest(query="Test", temperature=-0.1)
    
    def test_query_request_invalid_temperature_too_high(self):
        """Test that temperature > 2.0 is rejected"""
        with pytest.raises(ValidationError):
            QueryRequest(query="Test", temperature=2.1)


@pytest.mark.unit
class TestRetrievalResult:
    """Test suite for RetrievalResult schema"""
    
    def test_valid_retrieval_result(self):
        """Test creating a valid retrieval result"""
        result = RetrievalResult(
            rank=1,
            distance=0.234,
            branch="Disneyland_HongKong",
            rating=5.0,
            reviewer_location="Singapore",
            snippet="Great park!"
        )
        
        assert result.rank == 1
        assert result.distance == 0.234
        assert result.branch == "Disneyland_HongKong"
        assert result.rating == 5.0
    
    def test_retrieval_result_optional_rating(self):
        """Test that rating can be None"""
        result = RetrievalResult(
            rank=1,
            distance=0.234,
            branch="Disneyland_HongKong",
            rating=None,
            reviewer_location="Singapore",
            snippet="Great park!"
        )
        
        assert result.rating is None


@pytest.mark.unit
class TestQueryResponse:
    """Test suite for QueryResponse schema"""
    
    def test_valid_query_response(self):
        """Test creating a valid query response"""
        response = QueryResponse(
            query="Test query",
            answer="Test answer",
            k=5,
            model="gpt-4o-mini",
            temperature=0.2,
            retrieval_results=[
                RetrievalResult(
                    rank=1,
                    distance=0.234,
                    branch="Disneyland_HongKong",
                    rating=5.0,
                    reviewer_location="Singapore",
                    snippet="Great!"
                )
            ]
        )
        
        assert response.query == "Test query"
        assert response.answer == "Test answer"
        assert len(response.retrieval_results) == 1
    
    def test_query_response_empty_retrieval_results(self):
        """Test query response with no retrieval results"""
        response = QueryResponse(
            query="Test query",
            answer="Test answer",
            k=5,
            model="gpt-4o-mini",
            temperature=0.2,
            retrieval_results=[]
        )
        
        assert len(response.retrieval_results) == 0


@pytest.mark.unit
class TestHealthResponse:
    """Test suite for HealthResponse schema"""
    
    def test_valid_health_response(self):
        """Test creating a valid health response"""
        response = HealthResponse(
            status="healthy",
            app_name="Test App",
            version="1.0.0",
            index_loaded=True,
            total_vectors=1000
        )
        
        assert response.status == "healthy"
        assert response.index_loaded is True
        assert response.total_vectors == 1000

