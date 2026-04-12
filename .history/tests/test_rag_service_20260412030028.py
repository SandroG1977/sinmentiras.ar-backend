from app.core.config import settings
from app.services.rag_service import LocalRAGService
from app.services.rag_store import rag_store


def test_retrieve_returns_relevant_chunk(tmp_path) -> None:
    knowledge_file = tmp_path / "facts.md"
    knowledge_file.write_text(
        "Argentina has a federal political system.\n\n"
        "Inflation impacts household purchasing power.",
        encoding="utf-8",
    )

    old_enabled = settings.RAG_ENABLED
    old_paths = settings.RAG_KNOWLEDGE_PATHS

    try:
        settings.RAG_ENABLED = True
        settings.RAG_KNOWLEDGE_PATHS = [str(tmp_path)]

        rag = LocalRAGService()
        results = rag.retrieve("What impacts purchasing power?", top_k=2)

        assert results
        assert "purchasing power" in str(results[0]["text"]).lower()
    finally:
        settings.RAG_ENABLED = old_enabled
        settings.RAG_KNOWLEDGE_PATHS = old_paths


class _FakeResponse:
    def __init__(self, text: str, content: bytes | None = None) -> None:
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.apparent_encoding = "utf-8"
        self.encoding = "utf-8"

    def raise_for_status(self) -> None:
        return None


def test_ingest_law_from_infoleg_persists_article_chunks(monkeypatch, tmp_path) -> None:
    search_html = '<a href="/infolegInternet/verNorma.do?id=25552">Ley</a>'
    ver_norma_html = '<a href="/infolegInternet/anexos/25000-29999/25552/texact.htm">Texto actualizado</a>'
    texact_html = """
    <html>
      <head><title>Ley 20744</title></head>
      <body>
        <p>TITULO I</p>
        <p>CAPITULO I</p>
        <p>Art. 1 - Principios generales del contrato de trabajo.</p>
        <p>Establece la relacion de trabajo.</p>
        <p>Art. 2 - Ambito de aplicacion.</p>
        <p>Se aplica a todo contrato de trabajo.</p>
        <p>Nota Infoleg: Vigencia 01/01/1976 por sustitucion por Art. 2 bis (B.O. 01/01/1976)</p>
      </body>
    </html>
    """

    def fake_get(url: str, timeout: int):
        if "buscarNormas.do" in url:
            return _FakeResponse(search_html)
        if "verNorma.do" in url:
            return _FakeResponse(ver_norma_html)
        if "texact.htm" in url:
            return _FakeResponse(texact_html, texact_html.encode("utf-8"))
        raise AssertionError(f"Unexpected URL: {url}")

    old_db_path = settings.RAG_DB_PATH
    try:
        settings.RAG_DB_PATH = str(tmp_path / "rag_infoleg.sqlite3")
        monkeypatch.setattr("app.services.rag_service.requests.get", fake_get)

        rag = LocalRAGService()
        result = rag.ingest_law_from_infoleg(20744)

        assert result.law_id == 20744
        assert result.infoleg_id == 25552
        assert result.chunks_inserted >= 2

        persisted = rag_store.list_chunks()
        assert persisted
        assert persisted[0]["metadata"]["law_id"] == 20744
        assert any(row["metadata"].get("articulo_nro") == 1 for row in persisted)
        # Verify metadata structure is present even if notes count is 0 in this simple mock
        assert any(
            "cantidad_notas_substitucion" in row["metadata"] for row in persisted
        )
    finally:
        settings.RAG_DB_PATH = old_db_path
