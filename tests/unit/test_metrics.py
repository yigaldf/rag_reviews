"""Unit tests for MetricsCollector"""
import pytest
from app.utils.metrics import MetricsCollector


@pytest.mark.unit
@pytest.mark.metrics
class TestMetricsCollector:
    """Test suite for MetricsCollector class"""
    
    def test_initialization(self):
        """Test that metrics collector initializes correctly"""
        metrics = MetricsCollector()
        
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.total_response_time == 0.0
        assert len(metrics.response_times) == 0
        assert len(metrics.retrieval_distances) == 0
    
    def test_record_successful_request(self):
        """Test recording a successful request"""
        metrics = MetricsCollector()
        
        metrics.record_request(
            success=True,
            response_time=2.5,
            model="gpt-4o-mini",
            k=10,
            avg_retrieval_distance=0.35,
            query_length=50,
            response_length=200
        )
        
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.total_response_time == 2.5
        assert len(metrics.response_times) == 1
        assert metrics.response_times[0] == 2.5
        assert len(metrics.retrieval_distances) == 1
        assert metrics.retrieval_distances[0] == 0.35
    
    def test_record_failed_request(self):
        """Test recording a failed request"""
        metrics = MetricsCollector()
        
        metrics.record_request(
            success=False,
            response_time=1.0,
            model="gpt-4o-mini",
            k=5,
            error_type="timeout"
        )
        
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert metrics.error_types["timeout"] == 1
    
    def test_multiple_requests(self):
        """Test recording multiple requests"""
        metrics = MetricsCollector()
        
        for i in range(10):
            metrics.record_request(
                success=True,
                response_time=2.0 + i * 0.1,
                model="gpt-4o-mini",
                k=5,
                avg_retrieval_distance=0.3 + i * 0.01
            )
        
        assert metrics.total_requests == 10
        assert metrics.successful_requests == 10
        assert len(metrics.response_times) == 10
    
    def test_poor_retrieval_tracking(self):
        """Test that poor retrievals are tracked correctly"""
        metrics = MetricsCollector()
        
        # Good retrieval
        metrics.record_request(
            success=True,
            response_time=2.0,
            model="gpt-4o-mini",
            k=5,
            avg_retrieval_distance=0.3
        )
        
        # Poor retrieval (>0.5)
        metrics.record_request(
            success=True,
            response_time=2.0,
            model="gpt-4o-mini",
            k=5,
            avg_retrieval_distance=0.6
        )
        
        assert metrics.poor_retrieval_count == 1
    
    def test_get_stats_empty(self):
        """Test getting stats when no requests recorded"""
        metrics = MetricsCollector()
        stats = metrics.get_stats()
        
        assert stats["throughput"]["total_requests"] == 0
        assert stats["reliability"]["success_rate_percent"] == 0
        assert "latency" in stats
        assert "model_quality" in stats
    
    def test_get_stats_with_data(self):
        """Test getting stats with recorded data"""
        metrics = MetricsCollector()
        
        # Record some requests
        for i in range(5):
            metrics.record_request(
                success=True,
                response_time=2.0 + i * 0.5,
                model="gpt-4o-mini",
                k=10,
                avg_retrieval_distance=0.3 + i * 0.05,
                query_length=50,
                response_length=200
            )
        
        stats = metrics.get_stats()
        
        # Check basic counts
        assert stats["throughput"]["total_requests"] == 5
        assert stats["reliability"]["successful_requests"] == 5
        assert stats["reliability"]["success_rate_percent"] == 100.0
        
        # Check latency metrics
        assert "average_seconds" in stats["latency"]
        assert "p50_seconds" in stats["latency"]
        assert "p95_seconds" in stats["latency"]
        
        # Check model quality metrics
        assert "retrieval" in stats["model_quality"]
        assert "query_length" in stats["model_quality"]
        assert "response_length" in stats["model_quality"]
    
    def test_model_distribution(self):
        """Test tracking different models"""
        metrics = MetricsCollector()
        
        metrics.record_request(True, 2.0, "gpt-4o-mini", 5)
        metrics.record_request(True, 3.0, "gpt-4o-mini", 5)
        metrics.record_request(True, 4.0, "gpt-4o", 5)
        
        stats = metrics.get_stats()
        
        assert stats["usage"]["queries_by_model"]["gpt-4o-mini"] == 2
        assert stats["usage"]["queries_by_model"]["gpt-4o"] == 1
    
    def test_k_value_distribution(self):
        """Test tracking different k values"""
        metrics = MetricsCollector()
        
        metrics.record_request(True, 2.0, "gpt-4o-mini", 5)
        metrics.record_request(True, 2.0, "gpt-4o-mini", 10)
        metrics.record_request(True, 2.0, "gpt-4o-mini", 10)
        
        stats = metrics.get_stats()
        
        assert stats["usage"]["queries_by_k"][5] == 1
        assert stats["usage"]["queries_by_k"][10] == 2
    
    def test_memory_management(self):
        """Test that old data is discarded to prevent memory issues"""
        metrics = MetricsCollector()
        
        # Record 1500 requests (should keep only last 1000)
        for i in range(1500):
            metrics.record_request(
                success=True,
                response_time=2.0,
                model="gpt-4o-mini",
                k=5,
                avg_retrieval_distance=0.3
            )
        
        assert metrics.total_requests == 1500
        assert len(metrics.response_times) == 1000  # Should be capped
        assert len(metrics.retrieval_distances) == 1000  # Should be capped
    
    def test_reset(self):
        """Test resetting metrics"""
        metrics = MetricsCollector()
        
        # Record some data
        metrics.record_request(True, 2.0, "gpt-4o-mini", 5)
        assert metrics.total_requests == 1
        
        # Reset
        metrics.reset()
        
        # Verify everything is cleared
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert len(metrics.response_times) == 0
    
    def test_percentile_calculations(self):
        """Test that percentiles are calculated correctly"""
        metrics = MetricsCollector()
        
        # Record requests with known response times
        response_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for rt in response_times:
            metrics.record_request(True, rt, "gpt-4o-mini", 5)
        
        stats = metrics.get_stats()
        
        # Check percentiles are reasonable
        assert stats["latency"]["min_seconds"] == 1.0
        assert stats["latency"]["max_seconds"] == 10.0
        assert 4.0 <= stats["latency"]["p50_seconds"] <= 6.0  # Median should be ~5
        assert 8.0 <= stats["latency"]["p95_seconds"] <= 10.0

