from pydantic import BaseModel, Field, model_validator
from typing import Literal


class RAGIngestMinIORequest(BaseModel):
    minio_path: str | None = Field(default=None, min_length=1, max_length=2048)
    file_path: str | None = Field(default=None, min_length=1, max_length=4096)
    document_id: str | None = Field(default=None, max_length=255)
    presentado_por: str | None = Field(default=None, max_length=255)
    proyecto_tipo: Literal["base", "modificacion"] | None = None
    ley_base: str | None = Field(default=None, max_length=255)
    iniciado_en: str | None = Field(default=None, max_length=255)
    expediente_diputados: str | None = Field(default=None, max_length=255)
    expediente_senado: str | None = Field(default=None, max_length=255)
    publicado_en: str | None = Field(default=None, max_length=255)
    fecha: str | None = Field(default=None, max_length=64)
    ley_numero: str | None = Field(default=None, max_length=64)
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
        if self.proyecto_tipo == "modificacion" and not self.ley_base:
            raise ValueError(
                "ley_base is required when proyecto_tipo is 'modificacion'"
            )
        return self


class RAGIngestMinIOResponse(BaseModel):
    document_id: str
    source_uri: str
    chunks_inserted: int
    bytes_read: int
    db_path: str
