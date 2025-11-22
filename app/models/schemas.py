"""Pydantic schemas for API"""
from typing import List, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The question to ask", min_length=1)
    k: int = Field(5, description="Number of results to retrieve", ge=1, le=50)
    temperature: float = Field(0.2, description="LLM temperature", ge=0.0, le=2.0)
    model: str = Field("gpt-4o-mini", description="OpenAI model to use")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What do visitors like about Disneyland Hong Kong?",
                "k": 5,
                "temperature": 0.2,
                "model": "gpt-4o-mini"
            }
        }


class RetrievalResult(BaseModel):
    """Single retrieval result"""
    rank: int
    distance: float
    branch: str
    rating: Optional[float]
    reviewer_location: str
    snippet: str


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    query: str
    answer: str
    k: int
    model: str
    temperature: float
    retrieval_results: List[RetrievalResult]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    app_name: str
    version: str
    index_loaded: bool
    total_vectors: int
