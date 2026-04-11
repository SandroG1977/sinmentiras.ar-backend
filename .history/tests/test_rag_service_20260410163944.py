from app.core.config import settings
from app.services.rag_service import LocalRAGService


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
