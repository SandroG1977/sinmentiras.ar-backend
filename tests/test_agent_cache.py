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

    result, source = agent_graph.run_agent("consulta repetida")

    # Cache returned a plain string → falls back to _make_fallback_result
    assert isinstance(result, dict)
    assert result["summary_ia"] == "respuesta-cache"
    assert result["verdict"] == "SIN DATOS SUFICIENTES"
    assert source.startswith("cache:")
