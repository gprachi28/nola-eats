from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ROOT / ".env"))

    llm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    chroma_path: str = str(_ROOT / "chroma_db")
    chroma_collection: str = "yelp_reviews_nola"
    embed_model: str = "nomic-text-v1.5"
    embed_dimensions: int = 256
    sqlite_path: str = str(_ROOT / "yelp_reviews.db")
    session_ttl: int = 1800  # 30 minutes in seconds
    gemini_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_enabled: bool = True


settings = Settings()
