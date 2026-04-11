from hashlib import sha1
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.rag import RAGIngestMinIORequest, RAGIngestMinIOResponse
from app.services.document_text_service import document_text_service
from app.services.minio_service import minio_service
from app.services.rag_service import rag_service
from app.services.rag_store import rag_store
from app.services.text_chunker import chunk_text_by_article

router = APIRouter()


def _read_local_bytes(file_path: str) -> bytes:
    if not settings.LOCAL_FILE_INGEST_ENABLED:
        raise HTTPException(status_code=403, detail="Local file ingest is disabled")

    target = Path(file_path).expanduser().resolve()
    allowed_roots = [
        Path(base).expanduser().resolve()
        for base in settings.LOCAL_FILE_INGEST_BASE_PATHS
    ]
    if not any(root == target or root in target.parents for root in allowed_roots):
        raise HTTPException(
            status_code=403, detail="file_path is outside allowed ingest paths"
        )

    if not target.is_file():
        raise HTTPException(
            status_code=404, detail="file_path does not exist or is not a file"
        )

    try:
        return target.read_bytes()
    except OSError as exc:
        raise HTTPException(
            status_code=400, detail=f"Cannot read file_path: {exc}"
        ) from exc


@router.post("/ingest/minio", response_model=RAGIngestMinIOResponse)
def ingest_minio_document(payload: RAGIngestMinIORequest) -> RAGIngestMinIOResponse:
    source_uri = payload.minio_path or payload.file_path or ""
    if payload.minio_path:
        data = minio_service.read_bytes(payload.minio_path)
    else:
        data = _read_local_bytes(payload.file_path or "")

    bytes_read = len(data)

    text, used_ocr, source_file_type, ocr_reason = (
        document_text_service.extract_pure_text(
            data,
            source_uri,
        )
    )
    chunks = chunk_text_by_article(
        text,
        fallback_chunk_size=payload.chunk_size,
        fallback_chunk_overlap=payload.chunk_overlap,
    )

    if not chunks:
        return RAGIngestMinIOResponse(
            document_id=payload.document_id
            or sha1(source_uri.encode("utf-8")).hexdigest(),
            source_uri=source_uri,
            chunks_inserted=0,
            bytes_read=bytes_read,
            db_path=rag_store.db_path,
        )

    document_id = payload.document_id or sha1(source_uri.encode("utf-8")).hexdigest()
    inserted = rag_store.ingest_document(
        document_id=document_id,
        source_uri=source_uri,
        chunks=chunks,
        metadata={
            **(payload.metadata or {}),
            "used_ocr": str(used_ocr).lower(),
            "source_file_type": source_file_type,
            "ocr_reason": ocr_reason,
            "chunk_strategy": "article",
        },
        replace_existing=payload.replace_existing,
    )

    # Force a fresh in-memory index so new chunks are immediately retrievable.
    rag_service.refresh()

    return RAGIngestMinIOResponse(
        document_id=document_id,
        source_uri=source_uri,
        chunks_inserted=inserted,
        bytes_read=bytes_read,
        db_path=rag_store.db_path,
    )
