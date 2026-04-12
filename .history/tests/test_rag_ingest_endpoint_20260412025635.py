from fastapi.testclient import TestClient

from app.main import app
from app.services.minio_service import minio_service
from app.services.rag_service import InfolegIngestResult, rag_service
from app.services.rag_store import rag_store


client = TestClient(app)


class _FakeMinioService:
    def read_bytes(self, minio_path: str) -> bytes:
        assert minio_path == "minio://bucket/docs/context.txt"
        text = "SinMentiras es una plataforma de verificacion de hechos. " * 40
        return text.encode("utf-8")


class _FakeRAGService:
    def ingest_law_from_infoleg(
        self, law_id: int, *, replace_existing: bool = True
    ) -> InfolegIngestResult:
        assert law_id == 20744
        assert replace_existing is True
        return InfolegIngestResult(
            document_id="doc-20744",
            source_uri="https://servicios.infoleg.gob.ar/infolegInternet/anexos/25000-29999/25552/texact.htm",
            law_id=20744,
            infoleg_id=25552,
            chunks_inserted=12,
            source_norma="https://servicios.infoleg.gob.ar/infolegInternet/anexos/25000-29999/25552/norma.htm",
            source_actualizado="https://servicios.infoleg.gob.ar/infolegInternet/anexos/25000-29999/25552/texact.htm",
            sha256_hash="abc123",
        )


def test_ingest_minio_document_creates_chunks(tmp_path) -> None:
    original_db_path = rag_store.db_path

    try:
        from app.core.config import settings

        settings.RAG_DB_PATH = str(tmp_path / "rag_test.sqlite3")

        import app.api.v1.endpoints.rag as rag_endpoint

        rag_endpoint.minio_service = _FakeMinioService()  # type: ignore[assignment]

        response = client.post(
            "/api/v1/rag/ingest/minio",
            json={
                "minio_path": "minio://bucket/docs/context.txt",
                "chunk_size": 300,
                "chunk_overlap": 50,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["chunks_inserted"] > 1
        assert payload["source_uri"] == "minio://bucket/docs/context.txt"

        persisted = rag_store.list_chunks()
        assert persisted
        assert any("SinMentiras" in row["text"] for row in persisted)
    finally:
        from app.core.config import settings

        settings.RAG_DB_PATH = original_db_path
        import app.api.v1.endpoints.rag as rag_endpoint

        rag_endpoint.minio_service = minio_service  # type: ignore[assignment]


def test_ingest_infoleg_law_endpoint_returns_service_payload() -> None:
    import app.api.v1.endpoints.rag as rag_endpoint

    original_rag_service = rag_endpoint.rag_service
    try:
        rag_endpoint.rag_service = _FakeRAGService()  # type: ignore[assignment]

        response = client.post(
            "/api/v1/rag/ingest/law",
            json={"ley_numero": 20744, "replace_existing": True},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["law_id"] == 20744
        assert payload["infoleg_id"] == 25552
        assert payload["chunks_inserted"] == 12
        assert payload["source_actualizado"].endswith("texact.htm")
    finally:
        rag_endpoint.rag_service = original_rag_service  # type: ignore[assignment]
