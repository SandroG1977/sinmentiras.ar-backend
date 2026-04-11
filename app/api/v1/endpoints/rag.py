from datetime import datetime, timezone
from hashlib import sha1, sha256
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.rag import RAGIngestMinIORequest, RAGIngestMinIOResponse
from app.services.document_text_service import document_text_service
from app.services.legislative_metadata_service import legislative_metadata_service
from app.services.minio_service import minio_service
from app.services.rag_service import rag_service
from app.services.rag_store import rag_store
from app.services.text_chunker import chunk_text_by_article

router = APIRouter()


def _build_ingest_metadata(
    *,
    document_id: str,
    source_uri: str,
    bytes_read: int,
    source_file_type: str,
    used_ocr: bool,
    ocr_reason: str,
    chunk_count: int,
    chunk_strategy: str,
    source_sha256: str,
    presentado_por: str | None,
    proyecto_tipo: str | None,
    ley_base: str | None,
    iniciado_en: str | None,
    expediente_diputados: str | None,
    expediente_senado: str | None,
    publicado_en: str | None,
    fecha: str | None,
    ley_numero: str | None,
    user_metadata: dict[str, str | int | float | bool] | None,
) -> dict[str, str | int | float | bool]:
    source_kind = "minio" if source_uri.startswith("minio://") else "local"
    source_name = Path(source_uri).name if "/" in source_uri else source_uri

    normalized: dict[str, str | int | float | bool] = {
        "metadata_schema_version": "rag.v1",
        "document_id": document_id,
        "source_uri": source_uri,
        "source_kind": source_kind,
        "source_name": source_name,
        "source_file_type": source_file_type,
        "source_sha256": source_sha256,
        "bytes_read": bytes_read,
        "used_ocr": used_ocr,
        "ocr_reason": ocr_reason,
        "chunk_strategy": chunk_strategy,
        "chunk_count": chunk_count,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if presentado_por:
        normalized["presentado_por"] = presentado_por
    if proyecto_tipo:
        normalized["proyecto_tipo"] = proyecto_tipo
        normalized["es_modificacion"] = proyecto_tipo == "modificacion"
    if ley_base:
        normalized["ley_base"] = ley_base
    if iniciado_en:
        normalized["iniciado_en"] = iniciado_en
    if expediente_diputados:
        normalized["expediente_diputados"] = expediente_diputados
    if expediente_senado:
        normalized["expediente_senado"] = expediente_senado
    if publicado_en:
        normalized["publicado_en"] = publicado_en
    if fecha:
        normalized["fecha"] = fecha
    if ley_numero:
        normalized["ley_numero"] = ley_numero

    # Keep user-provided metadata namespaced to avoid collisions.
    for key, value in (user_metadata or {}).items():
        normalized[f"user_{key}"] = value

    return normalized


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
    source_sha256 = sha256(data).hexdigest()

    text, used_ocr, source_file_type, ocr_reason = (
        document_text_service.extract_pure_text(
            data,
            source_uri,
        )
    )
    inferred = legislative_metadata_service.infer_from_text(text)

    iniciado_en = payload.iniciado_en or inferred.get("iniciado_en")
    expediente_diputados = payload.expediente_diputados or inferred.get(
        "expediente_diputados"
    )
    expediente_senado = payload.expediente_senado or inferred.get("expediente_senado")
    publicado_en = payload.publicado_en or inferred.get("publicado_en")
    fecha = payload.fecha or inferred.get("fecha")
    ley_numero = payload.ley_numero or inferred.get("ley_numero")
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
    normalized_metadata = _build_ingest_metadata(
        document_id=document_id,
        source_uri=source_uri,
        bytes_read=bytes_read,
        source_file_type=source_file_type,
        used_ocr=used_ocr,
        ocr_reason=ocr_reason,
        chunk_count=len(chunks),
        chunk_strategy="article",
        source_sha256=source_sha256,
        presentado_por=payload.presentado_por,
        proyecto_tipo=payload.proyecto_tipo,
        ley_base=payload.ley_base,
        iniciado_en=iniciado_en,
        expediente_diputados=expediente_diputados,
        expediente_senado=expediente_senado,
        publicado_en=publicado_en,
        fecha=fecha,
        ley_numero=ley_numero,
        user_metadata=payload.metadata,
    )

    inserted = rag_store.ingest_document(
        document_id=document_id,
        source_uri=source_uri,
        chunks=chunks,
        metadata=normalized_metadata,
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
