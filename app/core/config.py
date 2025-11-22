"""Application configuration"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    app_name: str = "Disneyland Reviews RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # OpenAI Settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embed_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    # RAG Build Settings
    num_samples: int = 5000
    max_tokens: int = 500
    overlap: int = 50
    batch_size: int = 128
    
    # RAG Query Settings
    default_k: int = 5
    default_temperature: float = 0.2
    
    # Paths
    data_dir: Path = Path("data")
    index_dir: Path = Path("rag_index")
    log_dir: Path = Path("logs")
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
