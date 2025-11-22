"""Metrics collection for API monitoring"""
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


class MetricsCollector:
    """
    Comprehensive metrics collector for RAG system monitoring.
    
    Tracks key production metrics:
    - Latency: Response times and percentiles
    - Throughput: Requests per time period
    - Error Rate: Success/failure patterns
    - Model Quality: Retrieval quality and drift indicators
    """
    
    def __init__(self):
        # === REQUEST COUNTS ===
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # === LATENCY METRICS ===
        self.total_response_time = 0.0
        self.response_times: List[float] = []
        
        # === THROUGHPUT METRICS ===
        self.start_time = datetime.now()
        self.requests_per_minute: List[int] = []
        self.last_minute = datetime.now().minute
        self.current_minute_requests = 0
        
        # === ERROR TRACKING ===
        self.error_types: Dict[str, int] = defaultdict(int)
        
        # === MODEL QUALITY / DRIFT INDICATORS ===
        self.retrieval_distances: List[float] = []  # Track average distances
        self.poor_retrieval_count = 0  # Count of queries with high avg distance
        self.query_lengths: List[int] = []  # Track query length distribution
        self.response_lengths: List[int] = []  # Track response length distribution
        
        # === USAGE PATTERNS ===
        self.queries_by_model: Dict[str, int] = defaultdict(int)
        self.queries_by_k: Dict[int, int] = defaultdict(int)
        
        # === COST TRACKING ===
        self.total_tokens_used = 0  # If available from API
        
    def record_request(
        self, 
        success: bool, 
        response_time: float, 
        model: str, 
        k: int,
        error_type: Optional[str] = None,
        avg_retrieval_distance: Optional[float] = None,
        query_length: Optional[int] = None,
        response_length: Optional[int] = None,
        tokens_used: Optional[int] = None
    ):
        """
        Record a single request with comprehensive metadata.
        
        Args:
            success: Whether the request succeeded
            response_time: Total response time in seconds
            model: Model name used (e.g., 'gpt-4o-mini')
            k: Number of results retrieved
            error_type: Type of error if failed (e.g., 'timeout', 'api_error')
            avg_retrieval_distance: Average distance of retrieved results
            query_length: Length of user query in characters
            response_length: Length of generated response in characters
            tokens_used: Number of tokens consumed (if available)
        """
        # === REQUEST COUNTS ===
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_types[error_type] += 1
        
        # === LATENCY ===
        self.total_response_time += response_time
        self.response_times.append(response_time)
        
        # Keep only last 1000 response times to avoid memory issues
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        # === THROUGHPUT ===
        current_minute = datetime.now().minute
        if current_minute != self.last_minute:
            self.requests_per_minute.append(self.current_minute_requests)
            # Keep only last 60 minutes
            if len(self.requests_per_minute) > 60:
                self.requests_per_minute = self.requests_per_minute[-60:]
            self.current_minute_requests = 0
            self.last_minute = current_minute
        self.current_minute_requests += 1
        
        # === MODEL QUALITY / DRIFT ===
        if avg_retrieval_distance is not None:
            self.retrieval_distances.append(avg_retrieval_distance)
            # Keep only last 1000
            if len(self.retrieval_distances) > 1000:
                self.retrieval_distances = self.retrieval_distances[-1000:]
            
            # Flag poor retrievals (distance > 0.5 indicates weak relevance)
            if avg_retrieval_distance > 0.5:
                self.poor_retrieval_count += 1
        
        if query_length is not None:
            self.query_lengths.append(query_length)
            if len(self.query_lengths) > 1000:
                self.query_lengths = self.query_lengths[-1000:]
        
        if response_length is not None:
            self.response_lengths.append(response_length)
            if len(self.response_lengths) > 1000:
                self.response_lengths = self.response_lengths[-1000:]
        
        # === USAGE PATTERNS ===
        self.queries_by_model[model] += 1
        self.queries_by_k[k] += 1
        
        # === COST TRACKING ===
        if tokens_used is not None:
            self.total_tokens_used += tokens_used
    
    def get_stats(self) -> dict:
        """
        Get comprehensive statistics organized by category.
        
        Returns:
            dict: Nested dictionary with metrics organized by:
                - system: Uptime and basic info
                - latency: Response time metrics
                - throughput: Request volume over time
                - reliability: Error rates and patterns
                - model_quality: Retrieval quality and drift indicators
                - usage: Model and parameter distributions
                - cost: Token usage and cost estimates
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        avg_response_time = self.total_response_time / self.total_requests if self.total_requests > 0 else 0
        
        stats = {
            # === SYSTEM INFO ===
            "system": {
                "uptime_seconds": round(uptime, 2),
                "uptime_hours": round(uptime / 3600, 2),
                "start_time": self.start_time.isoformat(),
            },
            
            # === LATENCY METRICS ===
            "latency": {
                "average_seconds": round(avg_response_time, 3),
            },
            
            # === THROUGHPUT METRICS ===
            "throughput": {
                "total_requests": self.total_requests,
                "requests_per_second": round(self.total_requests / uptime, 2) if uptime > 0 else 0,
                "requests_per_minute": round(self.total_requests / (uptime / 60), 2) if uptime > 0 else 0,
                "current_minute_requests": self.current_minute_requests,
            },
            
            # === RELIABILITY / ERROR RATE ===
            "reliability": {
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate_percent": round(self.successful_requests / self.total_requests * 100, 2) if self.total_requests > 0 else 0,
                "error_rate_percent": round(self.failed_requests / self.total_requests * 100, 2) if self.total_requests > 0 else 0,
                "error_types": dict(self.error_types),
            },
            
            # === MODEL QUALITY / DRIFT INDICATORS ===
            "model_quality": {},
            
            # === USAGE PATTERNS ===
            "usage": {
                "queries_by_model": dict(self.queries_by_model),
                "queries_by_k": dict(self.queries_by_k),
            },
            
            # === COST METRICS ===
            "cost": {
                "total_tokens_used": self.total_tokens_used,
            }
        }
        
        # Add latency percentiles if we have data
        if self.response_times:
            sorted_times = sorted(self.response_times)
            n = len(sorted_times)
            stats["latency"].update({
                "min_seconds": round(min(sorted_times), 3),
                "max_seconds": round(max(sorted_times), 3),
                "p50_seconds": round(sorted_times[n // 2], 3),
                "p95_seconds": round(sorted_times[int(n * 0.95)], 3),
                "p99_seconds": round(sorted_times[int(n * 0.99)], 3),
            })
        
        # Add throughput history if available
        if self.requests_per_minute:
            stats["throughput"]["last_minute_requests"] = self.requests_per_minute[-1] if self.requests_per_minute else 0
            stats["throughput"]["avg_requests_per_minute"] = round(sum(self.requests_per_minute) / len(self.requests_per_minute), 2) if self.requests_per_minute else 0
            stats["throughput"]["peak_requests_per_minute"] = max(self.requests_per_minute) if self.requests_per_minute else 0
        
        # Add model quality metrics if available
        if self.retrieval_distances:
            avg_distance = sum(self.retrieval_distances) / len(self.retrieval_distances)
            sorted_distances = sorted(self.retrieval_distances)
            n = len(sorted_distances)
            
            stats["model_quality"]["retrieval"] = {
                "average_distance": round(avg_distance, 4),
                "min_distance": round(min(sorted_distances), 4),
                "max_distance": round(max(sorted_distances), 4),
                "p50_distance": round(sorted_distances[n // 2], 4),
                "p95_distance": round(sorted_distances[int(n * 0.95)], 4),
                "poor_retrieval_count": self.poor_retrieval_count,
                "poor_retrieval_rate_percent": round(self.poor_retrieval_count / len(self.retrieval_distances) * 100, 2),
            }
        
        # Add query length distribution
        if self.query_lengths:
            stats["model_quality"]["query_length"] = {
                "average": round(sum(self.query_lengths) / len(self.query_lengths), 1),
                "min": min(self.query_lengths),
                "max": max(self.query_lengths),
                "median": sorted(self.query_lengths)[len(self.query_lengths) // 2],
            }
        
        # Add response length distribution
        if self.response_lengths:
            stats["model_quality"]["response_length"] = {
                "average": round(sum(self.response_lengths) / len(self.response_lengths), 1),
                "min": min(self.response_lengths),
                "max": max(self.response_lengths),
                "median": sorted(self.response_lengths)[len(self.response_lengths) // 2],
            }
        
        # Add cost estimates (approximate, adjust based on your pricing)
        if self.total_tokens_used > 0:
            # Rough estimates for gpt-4o-mini: $0.15/1M input, $0.60/1M output
            # Assuming 50/50 split for simplicity
            estimated_cost = (self.total_tokens_used / 1_000_000) * 0.375
            stats["cost"]["estimated_cost_usd"] = round(estimated_cost, 4)
        
        return stats
    
    def reset(self):
        """Reset all metrics (useful for testing)"""
        self.__init__()

