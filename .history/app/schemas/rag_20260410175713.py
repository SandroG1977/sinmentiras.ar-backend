from pydantic import BaseModel, Field, model_validator


class RAGIngestMinIORequest(BaseModel):
    minio_path: str | None = Field(default=None, min_length=1, max_length=2048)
    file_path: str | None = Field(default=None, min_length=1, max_length=4096)
    document_id: str | None = Field(default=None, max_length=255)
    chunk_size: int = Field(default=700, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    metadata: dict[str, str | int | float | bool] | None = None
    replace_existing: bool = True

    @model_validator(mode="after")
    def validate_chunking(self) -> "RAGIngestMinIORequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if bool(self.minio_path) == bool(self.file_path):
            raise ValueError("Provide exactly one of minio_path or file_path")
        return self


class RAGIngestMinIOResponse(BaseModel):
    document_id: str
    source_uri: str
    chunks_inserted: int
    bytes_read: int
    db_path: str
