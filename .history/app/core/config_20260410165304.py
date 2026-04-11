from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    RAG_ENABLED: bool = True
    RAG_TOP_K: int = 3
    RAG_KNOWLEDGE_PATHS: list[str] = ["app/knowledge"]
    RAG_DB_PATH: str = "app/data/rag.sqlite3"

    MINIO_ENDPOINT: str | None = None
    MINIO_ACCESS_KEY: str | None = None
    MINIO_SECRET_KEY: str | None = None
    MINIO_SECURE: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
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


settings = Settings()
