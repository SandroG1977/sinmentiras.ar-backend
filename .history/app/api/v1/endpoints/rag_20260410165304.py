from hashlib import sha1

from fastapi import APIRouter

from app.schemas.rag import RAGIngestMinIORequest, RAGIngestMinIOResponse
from app.services.minio_service import minio_service
from app.services.rag_service import rag_service
from app.services.rag_store import rag_store
from app.services.text_chunker import chunk_text

router = APIRouter()


@router.post("/ingest/minio", response_model=RAGIngestMinIOResponse)
def ingest_minio_document(payload: RAGIngestMinIORequest) -> RAGIngestMinIOResponse:
    text, bytes_read = minio_service.read_text(payload.minio_path)
    chunks = chunk_text(
        text,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )

    if not chunks:
        return RAGIngestMinIOResponse(
            document_id=payload.document_id
            or sha1(payload.minio_path.encode("utf-8")).hexdigest(),
            source_uri=payload.minio_path,
            chunks_inserted=0,
            bytes_read=bytes_read,
            db_path=rag_store.db_path,
        )

    document_id = (
        payload.document_id or sha1(payload.minio_path.encode("utf-8")).hexdigest()
    )
    inserted = rag_store.ingest_document(
        document_id=document_id,
        source_uri=payload.minio_path,
        chunks=chunks,
        metadata=payload.metadata,
        replace_existing=payload.replace_existing,
    )

    # Force a fresh in-memory index so new chunks are immediately retrievable.
    rag_service.refresh()

    return RAGIngestMinIOResponse(
        document_id=document_id,
        source_uri=payload.minio_path,
        chunks_inserted=inserted,
        bytes_read=bytes_read,
        db_path=rag_store.db_path,
    )
