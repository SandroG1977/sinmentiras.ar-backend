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


class RAGIngestLawRequest(BaseModel):
    ley_numero: int = Field(ge=1)
    replace_existing: bool = True


class RAGIngestLawResponse(BaseModel):
    document_id: str
    source_uri: str
    law_id: int
    infoleg_id: int
    chunks_inserted: int
    source_norma: str
    source_actualizado: str
    sha256_hash: str
    db_path: str


class RAGAnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=3, ge=1, le=10)


class RAGSource(BaseModel):
    text: str
    source: str
    score: float


class RAGAnswerResponse(BaseModel):
    answer: str
    sources: list[RAGSource]
    context: str
    question: str


class RAGTruthIndexRequest(BaseModel):
    statement: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=3, ge=1, le=10)


class RAGTruthIndexResponse(BaseModel):
    indice_verdad: float | None
    justificacion: str
    statement: str
    sources: list[RAGSource]


class RAGLawStatusResponse(BaseModel):
    law_id: int
    exists: bool
    versions_loaded: int
    last_ingested_at: str | None = None
    latest_document_id: str | None = None
    latest_source_uri: str | None = None
    latest_source_actualizado: str | None = None
    latest_source_norma: str | None = None
    latest_sha256_hash: str | None = None
