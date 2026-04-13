from app.services import agent_graph


def test_run_agent_returns_cached_answer_without_invoking_llm(monkeypatch) -> None:
    monkeypatch.setattr(agent_graph.settings, "CACHE_QUERY_CHUNKING_ENABLED", False)
    monkeypatch.setattr(
        agent_graph.question_cache_store,
        "find_best_answer",
        lambda question, min_similarity: ("respuesta-cache", "mock", 0.991),
    )
    monkeypatch.setattr(
        agent_graph.agent_graph,
        "invoke",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("LLM should not run")
        ),
    )

    result, source, retrieved_chunks = agent_graph.run_agent("consulta repetida")

    # Cache returned a plain string → falls back to _make_fallback_result
    assert isinstance(result, dict)
    assert result["summary_ia"].lower().startswith("no se puede determinar")
    assert result["verdict"] == "SIN DATOS SUFICIENTES"
    assert source.startswith("cache:")
    assert isinstance(retrieved_chunks, list)


def test_normalize_summary_for_sin_datos_is_not_ambiguous() -> None:
    summary = (
        "No se puede determinar a partir del texto disponible. "
        "Sin embargo, esto implica que podría aplicar en algunos casos."
    )

    out = agent_graph._normalize_summary_for_verdict("SIN DATOS SUFICIENTES", summary)
    assert out.lower().startswith("no se puede determinar")
    assert "sin embargo" not in out.lower()


def test_normalize_summary_for_inconsistencia_removes_no_determinar_phrase() -> None:
    summary = "No se puede determinar con precisión, aunque hay conflicto técnico."

    out = agent_graph._normalize_summary_for_verdict("INCONSISTENCIA TÉCNICA", summary)
    assert "no se puede determinar" not in out.lower()
    assert "inconsistente" in out.lower()


def test_rerank_chunks_by_intent_prioritizes_working_time_articles() -> None:
    chunks = [
        {
            "text": "Art. 116. Concepto de salario mínimo vital...",
            "score": 0.95,
            "metadata": {
                "title": "Art. 116. —Concepto.",
                "path": "TITULO IV De la Remuneración",
                "articulo_nro": 116,
            },
        },
        {
            "text": "Art. 201. —Horas Suplementarias... recargo ...",
            "score": 0.55,
            "metadata": {
                "title": "Art. 201. —Horas Suplementarias",
                "path": "TITULO IX De la Duración del Trabajo y Descanso Semanal > CAPITULO I Jornada de Trabajo",
                "articulo_nro": 201,
            },
        },
    ]

    reranked = agent_graph._rerank_chunks_by_intent(
        "si mi jefe quiere puedo trabajar 12 horas por dia?",
        chunks,
    )

    assert reranked[0]["metadata"]["articulo_nro"] == 201


def test_rerank_chunks_by_intent_keeps_original_for_non_working_time_query() -> None:
    chunks = [
        {"text": "A", "score": 0.7, "metadata": {"articulo_nro": 10}},
        {"text": "B", "score": 0.6, "metadata": {"articulo_nro": 201}},
    ]

    reranked = agent_graph._rerank_chunks_by_intent("consulta sobre embargos", chunks)
    assert reranked == chunks


def test_prune_low_relevance_chunks_keeps_floor_when_scores_drop() -> None:
    chunks = [
        {"text": "a", "score": 1.0},
        {"text": "b", "score": 0.8},
        {"text": "c", "score": 0.5},
        {"text": "d", "score": 0.3},
        {"text": "e", "score": 0.2},
        {"text": "f", "score": 0.1},
    ]

    pruned = agent_graph._prune_low_relevance_chunks(
        chunks,
        min_keep=5,
        max_keep=8,
        relative_threshold=0.75,
    )

    assert len(pruned) == 5


def test_is_law_overview_query_detects_general_law_question() -> None:
    assert agent_graph._is_law_overview_query("de qué se trata la ley 20.744?")
    assert not agent_graph._is_law_overview_query("es falso que ahora son 12 horas?")


def test_extract_law_id_hints_supports_multiple_laws() -> None:
    prompt = "comparame la ley 20.744 y la ley 26.639 sobre este tema"
    assert agent_graph._extract_law_id_hints(prompt) == [20744, 26639]


def test_expand_law_id_hints_from_catalog_adds_keyword_matched_law(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_graph.rag_store,
        "search_law_catalog",
        lambda _prompt, limit=8: [
            {
                "law_id": 26639,
                "title": "Ley de glaciares y ambiente periglacial",
                "hash_tag": "#ley26639glaciares",
                "keywords": ["glaciares", "ambiente", "periglacial"],
            }
        ],
    )

    expanded = agent_graph._expand_law_id_hints_from_catalog(
        "necesito alcance de glaciares y ambiente periglacial",
        [],
    )

    assert expanded == [26639]


def test_rewrite_query_with_law_scope_uses_catalog_data(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_graph.rag_store,
        "get_law_catalog_entry",
        lambda law_id: {
            "law_id": law_id,
            "title": "Ley de glaciares",
            "keywords": ["glaciares", "ambiente", "reserva"],
        },
    )

    scoped = agent_graph._rewrite_query_with_law_scope("objeto y alcance", [26639])

    assert "Ley 26639" in scoped
    assert "Ley de glaciares" in scoped
    assert "glaciares" in scoped


def test_prioritize_chunks_for_law_supports_multiple_laws() -> None:
    chunks = [
        {"text": "a", "metadata": {"law_id": 20744, "articulo_nro": 199}},
        {"text": "b", "metadata": {"law_id": 12345, "articulo_nro": 1}},
        {"text": "c", "metadata": {"law_id": 26639, "articulo_nro": 6}},
    ]

    prioritized = agent_graph._prioritize_chunks_for_law(chunks, [20744, 26639])
    ids = [
        int(item["metadata"]["law_id"])
        for item in prioritized
        if isinstance(item.get("metadata"), dict)
    ]

    assert ids[:2] == [20744, 26639]


def test_augment_with_explicit_law_chunks_injects_missing_law(monkeypatch) -> None:
    existing = [
        {
            "text": "chunk de ley laboral",
            "source": "db",
            "score": 0.91,
            "metadata": {"law_id": 20744, "kind": "articulo", "articulo_ref": "199"},
        }
    ]

    monkeypatch.setattr(
        agent_graph.rag_store,
        "list_chunks",
        lambda: [
            {
                "text": "objeto y alcance de glaciares",
                "source": "db",
                "metadata": {"law_id": 26639, "kind": "articulo", "articulo_ref": "2"},
            },
            {
                "text": "jornada laboral limite",
                "source": "db",
                "metadata": {
                    "law_id": 20744,
                    "kind": "articulo",
                    "articulo_ref": "196",
                },
            },
        ],
    )

    merged = agent_graph._augment_with_explicit_law_chunks(
        "comparar objeto y alcance ley 20.744 y ley 26.639",
        existing,
        [20744, 26639],
        per_law=2,
    )

    ids = [agent_graph._metadata_law_id(c) for c in merged]
    assert 26639 in ids


# ---------------------------------------------------------------------------
# Cache-per-law tests (Phase 7)
# ---------------------------------------------------------------------------


def _make_cache_store(tmp_path, monkeypatch):
    """Returns a QuestionCacheStore pointed at a temporary SQLite DB."""
    from app.services.question_cache_store import QuestionCacheStore
    from app.core import config

    monkeypatch.setattr(
        config.settings, "CACHE_SQLITE_PATH", str(tmp_path / "cache_test.sqlite3")
    )
    monkeypatch.setattr(config.settings, "CACHE_POSTGRES_DSN", "")
    return QuestionCacheStore()


def test_question_cache_store_saves_law_ids_and_invalidates(monkeypatch, tmp_path):
    from app.services import embedding_service as emb_mod

    monkeypatch.setattr(
        emb_mod.embedding_service,
        "embed_texts",
        lambda texts: [[0.1] * 10 for _ in texts],
    )
    monkeypatch.setattr("app.core.config.settings.CACHE_ENABLED", True)

    store = _make_cache_store(tmp_path, monkeypatch)
    store.save_answer(
        "pregunta ley laboral", "respuesta A", "gpt-test", law_ids=[20744]
    )
    store.save_answer(
        "pregunta ley glaciares", "respuesta B", "gpt-test", law_ids=[26639]
    )

    deleted = store.invalidate_by_law_id(20744)

    assert deleted == 1

    # The entry for law 26639 must still be reachable via find_best_answer
    with store._connect() as conn:
        remaining = conn.execute("SELECT answer_text FROM question_cache").fetchall()

    assert len(remaining) == 1
    assert remaining[0][0] == "respuesta B"


def test_question_cache_store_invalidate_does_not_affect_other_laws(
    monkeypatch, tmp_path
):
    from app.services import embedding_service as emb_mod

    monkeypatch.setattr(
        emb_mod.embedding_service,
        "embed_texts",
        lambda texts: [[0.2] * 10 for _ in texts],
    )
    monkeypatch.setattr("app.core.config.settings.CACHE_ENABLED", True)

    store = _make_cache_store(tmp_path, monkeypatch)
    store.save_answer(
        "consulta sobre jornada", "respuesta laboral", "gpt-test", law_ids=[20744]
    )

    deleted = store.invalidate_by_law_id(99999)

    assert deleted == 0

    with store._connect() as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM question_cache").fetchone()

    assert remaining[0] == 1


def test_question_cache_store_invalidate_returns_zero_when_no_match(
    monkeypatch, tmp_path
):
    from app.services import embedding_service as emb_mod

    monkeypatch.setattr(
        emb_mod.embedding_service,
        "embed_texts",
        lambda texts: [[0.3] * 10 for _ in texts],
    )
    monkeypatch.setattr("app.core.config.settings.CACHE_ENABLED", True)

    store = _make_cache_store(tmp_path, monkeypatch)
    # Save with no law_ids
    store.save_answer("consulta genérica", "respuesta", "gpt-test", law_ids=None)

    deleted = store.invalidate_by_law_id(20744)

    assert deleted == 0


def test_overview_query_prepends_catalog_summary_to_context(monkeypatch) -> None:
    """Cuando es una overview query y hay summary_text en el catálogo, se inyecta antes del context_block."""

    captured_contexts: list[str] = []

    monkeypatch.setattr(agent_graph.settings, "CACHE_QUERY_CHUNKING_ENABLED", False)
    monkeypatch.setattr(agent_graph.settings, "RAG_ENABLED", True)
    monkeypatch.setattr(agent_graph.settings, "OPENAI_API_KEY", "sk-test")

    # No cache hit
    monkeypatch.setattr(
        agent_graph.question_cache_store,
        "find_best_answer",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        agent_graph.question_cache_store,
        "save_answer",
        lambda *_args, **_kwargs: None,
    )

    # Retrieval returns one chunk
    monkeypatch.setattr(
        agent_graph.rag_service,
        "retrieve",
        lambda *_args, **_kwargs: [
            {
                "text": "Artículo 1: objeto de la ley.",
                "source": "db",
                "score": 0.9,
                "metadata": {"law_id": 26639, "kind": "articulo", "articulo_ref": "1"},
            }
        ],
    )

    # Catalog has a pre-computed summary
    monkeypatch.setattr(
        agent_graph.rag_store,
        "get_law_catalog_entry",
        lambda law_id: {
            "law_id": law_id,
            "title": "Ley de Glaciares",
            "keywords": ["glaciares", "ambiente"],
            "summary_text": "Esta ley protege los glaciares y el ambiente periglacial.",
        },
    )
    monkeypatch.setattr(
        agent_graph.rag_store,
        "search_law_catalog",
        lambda *_args, **_kwargs: [],
    )

    def fake_invoke_structured(prompt, chunks):
        return (
            {
                "verdict": "SIN DATOS SUFICIENTES",
                "summary_ia": "resumen base",
                "source_law": "",
                "source_url": "",
                "original_text": "",
                "highlights": [],
                "news_context": [],
            },
            chunks,
        )

    monkeypatch.setattr(
        agent_graph, "_invoke_structured_result", fake_invoke_structured
    )

    def fake_hyde(prompt, context_block, verdict):
        captured_contexts.append(context_block)
        return None

    monkeypatch.setattr(agent_graph, "_generate_hyde_summary", fake_hyde)

    def fake_overview(prompt, context_block):
        captured_contexts.append(context_block)
        return "La ley 26639 protege los glaciares."

    monkeypatch.setattr(agent_graph, "_generate_law_overview_summary", fake_overview)

    result, _source, _chunks = agent_graph.run_agent("de qué se trata la ley 26.639?")

    # The context passed to the overview generator must contain the catalog summary
    assert any("Resumen oficial Ley 26639" in ctx for ctx in captured_contexts)
    assert result["summary_ia"] == "La ley 26639 protege los glaciares."
