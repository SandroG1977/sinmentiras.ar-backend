from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_agent_chat_includes_chunks_used(monkeypatch) -> None:
    def fake_run_agent(prompt: str):
        return (
            {
                "verdict": "VERDADERO",
                "summary_ia": "La jornada legal tiene límites y descansos obligatorios.",
                "source_law": "Ley 20.744 - Art. 197",
                "source_url": "https://servicios.infoleg.gob.ar/infolegInternet/anexos/25000-29999/25552/texact.htm",
                "original_text": "Distribución de la jornada y descanso entre jornadas.",
                "highlights": ["jornada", "descanso"],
                "news_context": [],
            },
            "mock",
            [
                {
                    "text": "Art. 197: distribución del tiempo de trabajo...",
                    "source": "https://servicios.infoleg.gob.ar/.../texact.htm",
                    "score": 0.9142,
                    "metadata": {
                        "chunk_index": 120,
                        "kind": "articulo",
                        "articulo_ref": "197",
                        "law_id": 20744,
                    },
                }
            ],
        )

    monkeypatch.setattr("app.api.v1.endpoints.chat.run_agent", fake_run_agent)

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "prompt": "Dicen que con la ley 20.744 voy a tener que trabajar 12 horas?",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert (
        data["query"]
        == "Dicen que con la ley 20.744 voy a tener que trabajar 12 horas?"
    )
    assert data["verdict"] == "VERDADERO"
    assert data["used_model"] == "mock"

    assert isinstance(data["chunks_used"], list)
    assert len(data["chunks_used"]) == 1

    first_chunk = data["chunks_used"][0]
    assert first_chunk["metadata"]["articulo_ref"] == "197"
    assert first_chunk["metadata"]["law_id"] == 20744
