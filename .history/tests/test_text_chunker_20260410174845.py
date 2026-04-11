from app.services.text_chunker import chunk_text_by_article


def test_chunk_text_by_article_splits_sections() -> None:
    text = (
        "Articulo 1. Objeto. Este articulo define el objeto de la ley.\n\n"
        "Articulo 2. Alcance. Este articulo define el alcance general."
    )

    chunks = chunk_text_by_article(text, fallback_chunk_size=200, fallback_chunk_overlap=40)

    assert len(chunks) == 2
    assert chunks[0].lower().startswith("articulo 1")
    assert chunks[1].lower().startswith("articulo 2")


def test_chunk_text_by_article_fallback_when_no_articles() -> None:
    text = "Texto sin articulos pero suficientemente largo para forzar fallback en chunks. " * 20

    chunks = chunk_text_by_article(text, fallback_chunk_size=120, fallback_chunk_overlap=20)

    assert len(chunks) > 1
