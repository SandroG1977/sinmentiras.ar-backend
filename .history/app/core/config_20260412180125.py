from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    APP_NAME: str = "sinmentiras.ar backend"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "sinmentiras-backend"

    RAG_ENABLED: bool = True
    RAG_TOP_K: int = 3
    RAG_KNOWLEDGE_PATHS: list[str] = ["app/knowledge"]
    RAG_DB_PATH: str = "app/data/rag.sqlite3"
    RAG_USE_FAISS: bool = True
    RAG_POSTGRES_DSN: str | None = None

    CACHE_ENABLED: bool = True
    CACHE_MIN_SIMILARITY: float = 0.93
    CACHE_SQLITE_PATH: str = "app/data/cache.sqlite3"
    CACHE_POSTGRES_DSN: str | None = None
    CACHE_QUERY_CHUNKING_ENABLED: bool = True
    CACHE_QUERY_CHUNK_SIZE: int = 240
    CACHE_QUERY_CHUNK_OVERLAP: int = 40

    MINIO_ENDPOINT: str | None = None
    MINIO_ACCESS_KEY: str | None = None
    MINIO_SECRET_KEY: str | None = None
    MINIO_SECURE: bool = False

    LOCAL_FILE_INGEST_ENABLED: bool = True
    LOCAL_FILE_INGEST_BASE_PATHS: list[str] = ["."]

    model_config = SettingsConfigDict(
        env_file=(str(APP_DIR / ".env"), str(APP_DIR.parent / ".env")),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("RAG_KNOWLEDGE_PATHS", mode="before")
    @classmethod
    def parse_rag_knowledge_paths(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [path.strip() for path in value.split(",") if path.strip()]
        return value

    @field_validator("LOCAL_FILE_INGEST_BASE_PATHS", mode="before")
    @classmethod
    def parse_local_file_ingest_base_paths(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [path.strip() for path in value.split(",") if path.strip()]
        return value


settings = Settings()
