from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_law_status_exists(monkeypatch) -> None:
    import app.api.v1.endpoints.rag as rag_endpoint

    def fake_status(_law_id: int) -> dict[str, object]:
        return {
            "law_id": 20744,
            "exists": True,
            "versions_loaded": 2,
            "last_ingested_at": "2026-04-12T15:35:00+00:00",
            "latest_document_id": "infoleg-25552-aa11bb22cc33",
            "latest_source_uri": "https://servicios.infoleg.gob.ar/infolegInternet/anexos/25000-29999/25552/texact.htm",
            "latest_source_actualizado": "https://servicios.infoleg.gob.ar/infolegInternet/anexos/25000-29999/25552/texact.htm",
            "latest_source_norma": "https://servicios.infoleg.gob.ar/infolegInternet/anexos/25000-29999/25552/norma.htm",
            "latest_sha256_hash": "abc123",
        }

    monkeypatch.setattr(rag_endpoint.rag_store, "get_latest_law_status", fake_status)

    response = client.get("/api/v1/rag/law/20744/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["law_id"] == 20744
    assert payload["exists"] is True
    assert payload["versions_loaded"] == 2
    assert payload["latest_sha256_hash"] == "abc123"


def test_law_status_not_found(monkeypatch) -> None:
    import app.api.v1.endpoints.rag as rag_endpoint

    def fake_status(_law_id: int) -> dict[str, object]:
        return {
            "law_id": 999999,
            "exists": False,
            "versions_loaded": 0,
            "last_ingested_at": None,
            "latest_document_id": None,
            "latest_source_uri": None,
            "latest_source_actualizado": None,
            "latest_source_norma": None,
            "latest_sha256_hash": None,
        }

    monkeypatch.setattr(rag_endpoint.rag_store, "get_latest_law_status", fake_status)

    response = client.get("/api/v1/rag/law/999999/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["exists"] is False
    assert payload["versions_loaded"] == 0
