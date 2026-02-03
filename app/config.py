from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # API Keys
    openai_api_key: str
    tavily_api_key: str
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "crag_documents"
    
    # OpenAI Models
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    embedding_dimensions: int = 1536
    
    # CRAG Settings
    crag_relevance_threshold: float = 0.7
    crag_ambiguous_threshold: float = 0.5
    
    # Self-Reflective Settings
    reflection_min_score: float = 0.8
    max_reflection_retries: int = 2
    
    # Retrieval
    top_k_results: int = 5
    
    # Upload
    upload_dir: str = "uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB


@lru_cache
def get_settings() -> Settings:
    return Settings()
