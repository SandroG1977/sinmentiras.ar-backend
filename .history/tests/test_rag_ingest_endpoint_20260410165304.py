from fastapi.testclient import TestClient

from app.main import app
from app.services.minio_service import minio_service
from app.services.rag_store import rag_store


client = TestClient(app)


class _FakeMinioService:
    def read_text(self, minio_path: str) -> tuple[str, int]:
        assert minio_path == "minio://bucket/docs/context.txt"
        text = "SinMentiras es una plataforma de verificacion de hechos. " * 40
        return text, len(text.encode("utf-8"))


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
