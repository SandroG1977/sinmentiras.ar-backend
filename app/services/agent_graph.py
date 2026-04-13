import dotenv
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.services.text_chunker import chunk_text
from app.services.question_cache_store import question_cache_store
from app.services.rag_store import rag_store
from app.services.rag_service import rag_service


dotenv.load_dotenv()  # Load environment variables from .env file, if present


def _configure_langsmith_tracing() -> None:
    # Ensure LangChain/LangSmith tracing picks up settings loaded from app config.
    if settings.LANGSMITH_TRACING:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
        if settings.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY


_configure_langsmith_tracing()

_SYSTEM_PROMPT = (
    "Eres un verificador legal para legislación argentina. "
    "Analiza el contexto proporcionado de documentos legales oficiales y responde de forma objetiva a la pregunta del usuario."
    "Tu respuesta sera ÚNICAMENTE con un objeto JSON válido (sin markdown, sin bloques de código) "
    "con exactamente estos campos:\n"
    "{\n"
    '  "verdict": "VERDADERO" | "FALSO" | "INCONSISTENCIA TÉCNICA" | "SIN DATOS SUFICIENTES",\n'
    '  "summary_ia": "análisis legal conciso en español (2-4 oraciones)",\n'
    '  "source_law": "ley y artículo específico citado, e.g. \'Ley 25.877 - Art. 24\'",\n'
    '  "source_url": "URL a la fuente oficial si se puede identificar, o cadena vacía",\n'
    '  "original_text": "fragmento textual más relevante del contexto",\n'
    '  "highlights": ["frase clave 1", "frase clave 2"],\n'
    '  "news_context": [{"source": "nombre del medio", "title": "título", "sentiment": "neutral|positive|negative"}]\n'
    "}\n"
    "Reglas obligatorias de calidad:\n"
    "1) Si el contexto contiene artículos relevantes, NO respondas de forma genérica ni digas que no hay datos.\n"
    "2) Da una conclusión clara para la pregunta del usuario (sí/no/depende) y luego explica por qué.\n"
    "3) Cita en source_law la ley y artículo más relevante cuando exista en el contexto.\n"
    "4) Mantén summary_ia útil para usuario final, precisa y en lenguaje claro.\n"
    "No incluyas ningún texto fuera del JSON."
)

# ---------------------------------------------------------------------------
# Rewriter: reformula la pregunta coloquial en una consulta optimizada para
# el vector store antes del retrieval (mismo rol que rewriter_pipeline del notebook).
# ---------------------------------------------------------------------------
_REWRITER_PROMPT = PromptTemplate.from_template(
    "Tu tarea es reformular la siguiente pregunta del usuario para optimizar "
    "la búsqueda semántica en una base vectorial de leyes argentinas. "
    "Devuelve ÚNICAMENTE la consulta reformulada, sin comillas, sin explicación, "
    "sin texto adicional.\n\n"
    "Pregunta del usuario: {pregunta}\n\n"
    "Consulta optimizada para búsqueda vectorial:"
)

# ---------------------------------------------------------------------------
# HYDE: genera una explicación anclada SOLO en el contexto recuperado,
# sin tecnicismos, para reemplazar summary_ia con una respuesta grounded.
# (mismo rol que hyde_pipeline del notebook)
# ---------------------------------------------------------------------------
_HYDE_PROMPT = PromptTemplate.from_template(
    "Actuando como Asesor Jurídico y Legislativo: responde la siguiente pregunta "
    "en al menos 2 párrafos, en lenguaje claro sin tecnicismos legales para que "
    "sea entendible por cualquier persona, utilizando ÚNICAMENTE el contexto "
    "normativo proporcionado como fuente de verdad.\n\n"
    "Debes respetar estrictamente este veredicto preliminar: {verdict}.\n"
    "- Si el veredicto es 'SIN DATOS SUFICIENTES', tu respuesta NO puede incluir "
    "conclusiones afirmativas/negativas ni recomendaciones condicionales. "
    "Debe limitarse a explicar por qué no se puede determinar con el contexto disponible.\n"
    "- Si el veredicto NO es 'SIN DATOS SUFICIENTES', no uses la frase 'no se puede determinar'.\n\n"
    "Si el contexto no contiene información suficiente para responder la pregunta, "
    "indica explícitamente que no se puede determinar a partir del texto disponible. "
    "Evita cualquier interpretación o suposición que no esté sustentada textualmente.\n\n"
    "Contexto normativo:\n{contexto}\n\n"
    "Pregunta: {pregunta}"
)

_LAW_OVERVIEW_PROMPT = PromptTemplate.from_template(
    "Explica de forma simple y breve de qué trata la ley usando SOLO el contexto "
    "proporcionado.\n"
    "- Escribe en español claro para público general.\n"
    "- Máximo 4 oraciones.\n"
    "- Incluye alcance general y 2-3 temas principales del texto.\n"
    "- Si el contexto es insuficiente, dilo explícitamente.\n\n"
    "Contexto:\n{contexto}\n\n"
    "Consulta del usuario: {pregunta}"
)


class AgentState(TypedDict):
    prompt: str
    output: str
    retrieved_chunks: list[dict[str, object]]
    context_chunks: list[str]
    structured_result: dict


def _make_fallback_result(summary: str) -> dict:
    return {
        "verdict": "SIN DATOS SUFICIENTES",
        "summary_ia": summary,
        "source_law": "",
        "source_url": "",
        "original_text": "",
        "highlights": [],
        "news_context": [],
    }


def _build_context_block(chunks: list[dict[str, object]]) -> str:
    """Builds the formatted context string passed to both verdict and HYDE prompts."""
    return "\n\n".join(
        f"[{idx}] Source: {item['source']}\n{item['text']}"
        for idx, item in enumerate(chunks, start=1)
        if isinstance(item, dict)
    )


def _rewrite_query(prompt: str) -> str:
    """Reformulates a colloquial/informal question into an optimized vector-DB search query.
    Falls back to the original prompt on any failure or when API key is absent.
    """
    if not settings.RAG_REWRITER_ENABLED or not settings.OPENAI_API_KEY:
        return prompt
    try:
        llm = ChatOpenAI(
            model=settings.OPENAI_QUERY_MODEL,
            api_key=lambda: settings.OPENAI_API_KEY or "",
            temperature=0,
        )
        pipeline = _REWRITER_PROMPT | llm | StrOutputParser()
        rewritten = pipeline.invoke({"pregunta": prompt}).strip(' "\n\t')
        return rewritten if rewritten else prompt
    except Exception:
        return prompt


def _generate_hyde_summary(
    original_prompt: str,
    context_block: str,
    verdict: str,
) -> str | None:
    """Generates a human-readable, grounded explanation using retrieved chunks as the
    sole source of truth. Returns None on failure so the caller can keep the original.
    """
    if (
        not settings.RAG_HYDE_ENABLED
        or not context_block.strip()
        or not settings.OPENAI_API_KEY
    ):
        return None
    try:
        llm = ChatOpenAI(
            model=settings.OPENAI_QUERY_MODEL,
            api_key=lambda: settings.OPENAI_API_KEY or "",
            temperature=0,
        )
        pipeline = _HYDE_PROMPT | llm | StrOutputParser()
        result = pipeline.invoke(
            {
                "pregunta": original_prompt,
                "contexto": context_block,
                "verdict": verdict,
            }
        )
        return result.strip() if result.strip() else None
    except Exception:
        return None


def _normalize_summary_for_verdict(verdict: str, summary: str) -> str:
    normalized = (summary or "").strip()
    if not normalized:
        return normalized

    low = normalized.lower()

    if verdict == "SIN DATOS SUFICIENTES":
        return (
            "No se puede determinar con certeza a partir del texto legal recuperado. "
            "El contexto disponible no alcanza para concluir si esa obligación específica aplica en tu caso."
        )

    if verdict == "VERDADERO" and "no se puede determinar" in low:
        return (
            "Sí. Según el contexto legal recuperado, la afirmación resulta verdadera "
            "en los términos consultados."
        )

    if verdict == "FALSO" and "no se puede determinar" in low:
        return (
            "No. Según el contexto legal recuperado, la afirmación resulta falsa "
            "en los términos consultados."
        )

    if verdict == "INCONSISTENCIA TÉCNICA" and "no se puede determinar" in low:
        return (
            "La afirmación es técnicamente inconsistente según el contexto legal recuperado. "
            "Combina una conclusión tajante con información normativa incompleta o descontextualizada."
        )

    return normalized


def _build_graph():
    def run_model(state: AgentState) -> AgentState:
        prompt = state["prompt"]
        context_chunks: list[str] = []
        retrieved_chunks = state.get("retrieved_chunks") or []
        retrieved = retrieved_chunks

        if settings.RAG_ENABLED:
            retrieved = retrieved_chunks or rag_service.retrieve(
                prompt,
                top_k=settings.RAG_TOP_K,
            )
            if retrieved:
                context_chunks = [
                    f"[{idx}] Source: {item['source']}\n{item['text']}"
                    for idx, item in enumerate(retrieved, start=1)
                ]
                context_block = "\n\n".join(context_chunks)
                user_content = f"Context from official documents:\n{context_block}\n\nUser question: {prompt}"
            else:
                user_content = prompt
        else:
            user_content = prompt

        # Fallback to a deterministic mock response when no API key is set.
        if not settings.OPENAI_API_KEY:
            mock_output = f"[mock-response] {prompt}"
            return {
                "prompt": prompt,
                "output": mock_output,
                "retrieved_chunks": retrieved,
                "context_chunks": context_chunks,
                "structured_result": _make_fallback_result(mock_output),
            }

        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=lambda: settings.OPENAI_API_KEY or "",
            temperature=settings.OPENAI_RESPONSE_TEMPERATURE,
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            structured = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            structured = _make_fallback_result(content)

        return {
            "prompt": prompt,
            "output": content,
            "retrieved_chunks": retrieved,
            "context_chunks": context_chunks,
            "structured_result": structured,
        }

    graph = StateGraph(AgentState)
    graph.add_node("run_model", run_model)
    graph.add_edge(START, "run_model")
    graph.add_edge("run_model", END)
    return graph.compile()


agent_graph = _build_graph()


def make_audit_id() -> str:
    year = datetime.now(timezone.utc).year
    short = uuid.uuid4().hex[:6].upper()
    return f"AUD-{year}-{short}"


def make_response_hash(query: str, summary: str) -> str:
    raw = f"{query}|{summary}"
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def _build_query_chunks(prompt: str) -> list[str]:
    query_chunks = [prompt]
    if not settings.CACHE_QUERY_CHUNKING_ENABLED:
        return query_chunks

    chunked = chunk_text(
        prompt,
        chunk_size=settings.CACHE_QUERY_CHUNK_SIZE,
        chunk_overlap=settings.CACHE_QUERY_CHUNK_OVERLAP,
    )
    query_chunks.extend([chunk for chunk in chunked if chunk and chunk != prompt])
    return query_chunks


def _extract_law_id_hints(prompt: str) -> list[int]:
    matches = re.finditer(
        r"\bley\s*(?:n(?:ro)?\.?|n°)?\s*(\d{1,3}(?:[\.,]\d{3})+|\d{4,6})\b",
        prompt,
        flags=re.IGNORECASE,
    )

    law_ids: list[int] = []
    seen: set[int] = set()
    for match in matches:
        normalized = re.sub(r"[\.,]", "", match.group(1))
        try:
            law_id = int(normalized)
        except ValueError:
            continue
        if law_id <= 0 or law_id in seen:
            continue
        seen.add(law_id)
        law_ids.append(law_id)

    return law_ids


def _expand_law_id_hints_from_catalog(
    prompt: str, law_id_hints: list[int]
) -> list[int]:
    """Completa law_ids a partir del catalogo usando titulo/hashtag/keywords."""

    seen: set[int] = set(law_id_hints)
    merged: list[int] = list(law_id_hints)

    try:
        candidates: list[dict[str, object]] = []
        seen_candidate_ids: set[int] = set()

        # 1) intento directo con el prompt completo
        for candidate in rag_store.search_law_catalog(prompt, limit=8):
            cid = candidate.get("law_id")
            if isinstance(cid, int) and cid not in seen_candidate_ids:
                seen_candidate_ids.add(cid)
                candidates.append(candidate)

        # 2) fallback por términos para consultas largas (ej: "de qué se trata la ley de glaciares")
        stop_terms = {
            "de",
            "que",
            "qué",
            "trata",
            "ley",
            "sobre",
            "donde",
            "cómo",
            "como",
            "para",
            "esta",
            "este",
        }
        prompt_terms = [
            term
            for term in re.findall(r"[a-záéíóúñ0-9]{4,}", prompt.lower())
            if term not in stop_terms
        ]
        for term in prompt_terms[:6]:
            for candidate in rag_store.search_law_catalog(term, limit=8):
                cid = candidate.get("law_id")
                if isinstance(cid, int) and cid not in seen_candidate_ids:
                    seen_candidate_ids.add(cid)
                    candidates.append(candidate)
    except Exception:
        candidates = []

    prompt_low = prompt.lower()
    prompt_terms = set(re.findall(r"[a-záéíóúñ0-9]{4,}", prompt_low))
    # Para consultas generales del tipo "ley de glaciares", una sola coincidencia
    # temática fuerte suele ser suficiente para identificar la norma objetivo.
    is_general_law_topic_query = "ley" in prompt_low and not any(
        char.isdigit() for char in prompt_low
    )
    min_overlap = 1 if is_general_law_topic_query else 2
    for candidate in candidates:
        law_id_value = candidate.get("law_id")
        if not isinstance(law_id_value, int) or law_id_value in seen:
            continue

        title = str(candidate.get("title", "")).lower()
        hash_tag = str(candidate.get("hash_tag", "")).lower()
        keywords = candidate.get("keywords")
        keywords_text = " ".join(keywords) if isinstance(keywords, list) else ""
        summary_text = str(candidate.get("summary_text", "")).lower()
        haystack = f"{title} {hash_tag} {keywords_text} {summary_text}".lower()

        overlap = sum(1 for term in prompt_terms if term in haystack)
        if overlap >= min_overlap:
            seen.add(law_id_value)
            merged.append(law_id_value)

    return merged


def _rewrite_query_with_law_scope(search_query: str, law_id_hints: list[int]) -> str:
    """Rearma la consulta para sesgar retrieval a las leyes detectadas."""

    if not law_id_hints:
        return search_query

    law_refs: list[str] = []
    for law_id in law_id_hints[:5]:
        catalog = rag_store.get_law_catalog_entry(law_id)
        if isinstance(catalog, dict):
            title = str(catalog.get("title", "")).strip()
            keywords = catalog.get("keywords")
            keywords_text = (
                ", ".join(keywords[:4]) if isinstance(keywords, list) else ""
            )
            ref = f"Ley {law_id}"
            if title:
                ref += f" ({title})"
            if keywords_text:
                ref += f" [{keywords_text}]"
            law_refs.append(ref)
        else:
            law_refs.append(f"Ley {law_id}")

    if not law_refs:
        return search_query

    scope = "; ".join(law_refs)
    return (
        f"{search_query}. "
        f"Ambito normativo objetivo: {scope}. "
        "Priorizar articulos de esas leyes para responder."
    )


def _metadata_law_id(chunk: dict[str, object]) -> int | None:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("law_id")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _prioritize_chunks_for_law(
    chunks: list[dict[str, object]],
    law_id_hints: list[int],
) -> list[dict[str, object]]:
    if not law_id_hints:
        return chunks

    allowed = set(law_id_hints)
    matching = [chunk for chunk in chunks if _metadata_law_id(chunk) in allowed]
    non_matching = [chunk for chunk in chunks if _metadata_law_id(chunk) not in allowed]
    if matching:
        return matching + non_matching
    return chunks


def _augment_with_explicit_law_chunks(
    query: str,
    chunks: list[dict[str, object]],
    law_id_hints: list[int],
    per_law: int = 4,
) -> list[dict[str, object]]:
    """Asegura cobertura de leyes explícitamente mencionadas en la consulta."""

    if not law_id_hints or per_law <= 0:
        return chunks

    hinted = set(law_id_hints)
    query_terms = _extract_query_terms(query)

    def law_lexical_score(row: dict[str, object]) -> float:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        text = str(row.get("text", "")).lower()
        title = str(metadata.get("title", "")).lower()
        path = str(metadata.get("path", "")).lower()
        combined = f"{title} {path} {text}"

        score = 0.0
        for term in query_terms:
            if term in title:
                score += 4.0
            elif term in path:
                score += 2.0
            elif term in combined:
                score += 1.0

        if metadata.get("kind") == "articulo":
            score += 1.0
        return score

    per_law_candidates: dict[int, list[dict[str, object]]] = {
        law_id: [] for law_id in hinted
    }
    for row in rag_store.list_chunks():
        law_id = _metadata_law_id(row)
        if law_id not in hinted:
            continue
        per_law_candidates[law_id].append(row)

    injected: list[dict[str, object]] = []
    for law_id in law_id_hints:
        law_rows = per_law_candidates.get(law_id) or []
        if not law_rows:
            continue
        ordered = sorted(law_rows, key=law_lexical_score, reverse=True)
        for row in ordered[:per_law]:
            injected.append(
                {
                    "text": str(row.get("text", "")),
                    "source": str(row.get("source", "db")),
                    "score": round(0.75 + law_lexical_score(row) / 100.0, 4),
                    "metadata": (
                        row.get("metadata")
                        if isinstance(row.get("metadata"), dict)
                        else {}
                    ),
                }
            )

    merged = injected + chunks
    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str, int | None, str | None]] = set()
    for chunk in merged:
        metadata = (
            chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        )
        key = (
            str(chunk.get("source", "")),
            str(chunk.get("text", "")),
            _metadata_law_id(chunk),
            (
                str(metadata.get("articulo_ref"))
                if metadata.get("articulo_ref") is not None
                else None
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)

    return deduped


_STOPWORDS_ES = {
    "que",
    "como",
    "para",
    "porque",
    "donde",
    "cuando",
    "sobre",
    "entre",
    "desde",
    "hasta",
    "ahora",
    "tengo",
    "tiene",
    "puede",
    "puedo",
    "debo",
    "deber",
    "deberia",
    "debería",
    "quiero",
    "quiera",
    "jefe",
    "patron",
    "patrón",
    "trabajo",
    "trabajar",
    "dia",
    "día",
    "dias",
    "días",
    "ley",
    "art",
    "articulo",
    "artículo",
}


def _extract_query_terms(prompt: str) -> list[str]:
    tokens = re.findall(r"[a-záéíóúñ0-9]{3,}", prompt.lower())
    terms = [t for t in tokens if t not in _STOPWORDS_ES]
    seen: set[str] = set()
    unique_terms: list[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        unique_terms.append(t)
    return unique_terms


def _semantic_chunk_boost(chunk: dict[str, object], query_terms: list[str]) -> int:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    text = str(chunk.get("text", "")).lower()
    title = str(metadata.get("title", "")).lower()
    path = str(metadata.get("path", "")).lower()

    score = 0

    # Priorización estructural general (aplicable a cualquier ley cargada).
    if metadata.get("kind") == "articulo":
        score += 12
    if "capitulo" in path or "seccion" in path or "título" in path or "titulo" in path:
        score += 6

    # Matching semántico query->chunk (sin hardcode por ley/artículo).
    for term in query_terms:
        if term in title:
            score += 14
        if term in path:
            score += 8
        if term in text:
            score += 3

    # Penaliza chunks con baja señal semántica cuando hay muchos candidatos.
    if query_terms and not any(
        t in (title + " " + path + " " + text) for t in query_terms
    ):
        score -= 10

    return score


def _rerank_chunks_by_intent(
    prompt: str, chunks: list[dict[str, object]]
) -> list[dict[str, object]]:
    if not chunks:
        return chunks
    query_terms = _extract_query_terms(prompt)
    if not query_terms:
        return chunks

    # Re-ranking estable: boost semántico + score vectorial original como desempate.
    return sorted(
        chunks,
        key=lambda c: (
            _semantic_chunk_boost(c, query_terms),
            float(c.get("score") or 0.0),
        ),
        reverse=True,
    )


def _prune_low_relevance_chunks(
    chunks: list[dict[str, object]],
    min_keep: int,
    max_keep: int,
    relative_threshold: float = 0.75,
) -> list[dict[str, object]]:
    if not chunks:
        return []

    ordered = chunks[:max_keep]
    if len(ordered) <= min_keep:
        return ordered

    top_score = float(ordered[0].get("score") or 0.0)
    if top_score <= 0:
        return ordered

    min_score = top_score * relative_threshold
    kept = [c for c in ordered if float(c.get("score") or 0.0) >= min_score]

    if len(kept) < min_keep:
        return ordered[:min_keep]
    return kept[:max_keep]


def _is_law_overview_query(prompt: str) -> bool:
    p = prompt.lower()
    patterns = [
        r"de\s+que\s+se\s+trata\s+la\s+ley",
        r"de\s+qué\s+se\s+trata\s+la\s+ley",
        r"de\s+que\s+trata\s+la\s+ley",
        r"de\s+qué\s+trata\s+la\s+ley",
        r"resumen\s+de\s+la\s+ley",
        r"explic(?:a|ame)\s+la\s+ley",
        r"que\s+dice\s+la\s+ley\s+en\s+general",
        r"qué\s+dice\s+la\s+ley\s+en\s+general",
    ]
    return any(re.search(pattern, p) for pattern in patterns)


def _generate_law_overview_summary(prompt: str, context_block: str) -> str | None:
    if not context_block.strip() or not settings.OPENAI_API_KEY:
        return None
    try:
        llm = ChatOpenAI(
            model=settings.OPENAI_QUERY_MODEL,
            api_key=lambda: settings.OPENAI_API_KEY or "",
            temperature=settings.OPENAI_RESPONSE_TEMPERATURE,
        )
        pipeline = _LAW_OVERVIEW_PROMPT | llm | StrOutputParser()
        result = pipeline.invoke({"pregunta": prompt, "contexto": context_block})
        cleaned = result.strip()
        return cleaned if cleaned else None
    except Exception:
        return None


def _find_best_cached(query_chunks: list[str]) -> tuple[str, str, float] | None:
    best_cached: tuple[str, str, float] | None = None
    for query_piece in query_chunks:
        cached = question_cache_store.find_best_answer(
            question=query_piece,
            min_similarity=settings.CACHE_MIN_SIMILARITY,
        )
        if not cached:
            continue
        if best_cached is None or cached[2] > best_cached[2]:
            best_cached = cached
    return best_cached


def _parse_cached_answer(cached_answer: str) -> dict:
    try:
        parsed = json.loads(cached_answer)
        return (
            parsed if isinstance(parsed, dict) else _make_fallback_result(cached_answer)
        )
    except (json.JSONDecodeError, ValueError):
        return _make_fallback_result(cached_answer)


def _invoke_structured_result(
    prompt: str,
    retrieved_chunks: list[dict[str, object]],
) -> tuple[dict, list[dict[str, object]]]:
    result = agent_graph.invoke(
        {
            "prompt": prompt,
            "output": "",
            "retrieved_chunks": retrieved_chunks,
            "context_chunks": [],
            "structured_result": {},
        }
    )
    structured = result.get("structured_result")
    used_chunks = result.get("retrieved_chunks")
    normalized_chunks = (
        used_chunks if isinstance(used_chunks, list) else retrieved_chunks
    )
    if isinstance(structured, dict):
        return structured, normalized_chunks
    return _make_fallback_result(result.get("output", "")), normalized_chunks


def _cache_structured_answer(
    prompt: str,
    structured: dict,
    source: str,
    law_ids: list[int] | None = None,
) -> None:
    question_cache_store.save_answer(
        question=prompt,
        answer=json.dumps(structured, ensure_ascii=False),
        used_model=source,
        law_ids=law_ids or [],
    )


def run_agent(
    prompt: str, top_k: int | None = None
) -> tuple[dict, str, list[dict[str, object]]]:
    effective_top_k = max(top_k if top_k is not None else settings.RAG_TOP_K, 5)
    is_overview_query = _is_law_overview_query(prompt)
    law_id_hints = _extract_law_id_hints(prompt)
    law_id_hints = _expand_law_id_hints_from_catalog(prompt, law_id_hints)

    # ── Step 1: Rewriter ─────────────────────────────────────────────────────
    # Reformulate the user's colloquial question into a semantically-precise
    # query before hitting the vector store (mirrors rewriter_pipeline del notebook).
    search_query = _rewrite_query(prompt)
    search_query = _rewrite_query_with_law_scope(search_query, law_id_hints)

    # ── Step 2: Retrieval (with law-priority override) ────────────────────────
    retrieved_chunks = (
        _prioritize_chunks_for_law(
            rag_service.retrieve(search_query, top_k=effective_top_k * 3),
            law_id_hints,
        )
        if settings.RAG_ENABLED
        else []
    )
    retrieved_chunks = _augment_with_explicit_law_chunks(
        search_query,
        retrieved_chunks,
        law_id_hints,
        per_law=4,
    )

    # For "de que se trata la ley ..." queries, avoid cross-law pollution:
    # if we identified target laws, keep only chunks from those laws.
    if is_overview_query and law_id_hints:
        allowed = set(law_id_hints)
        filtered_chunks = [
            chunk for chunk in retrieved_chunks if _metadata_law_id(chunk) in allowed
        ]
        if filtered_chunks:
            retrieved_chunks = filtered_chunks

    retrieved_chunks = _rerank_chunks_by_intent(prompt, retrieved_chunks)
    retrieved_chunks = _prune_low_relevance_chunks(
        retrieved_chunks,
        min_keep=min(5, effective_top_k),
        max_keep=effective_top_k,
        relative_threshold=0.75,
    )

    # ── Step 3: Cache check (bypassed for explicit law queries) ───────────────
    # Cache uses the *original* prompt so semantically-equivalent colloquial
    # questions still get cache hits regardless of how they were rewritten.
    query_chunks = _build_query_chunks(prompt)
    # Overview prompts should bypass cache to avoid reusing old generic summaries.
    best_cached = (
        None if law_id_hints or is_overview_query else _find_best_cached(query_chunks)
    )

    if best_cached:
        cached_answer_str, cached_model, _similarity = best_cached
        cached_structured = _parse_cached_answer(cached_answer_str)
        cached_structured["summary_ia"] = _normalize_summary_for_verdict(
            str(cached_structured.get("verdict", "SIN DATOS SUFICIENTES")),
            str(cached_structured.get("summary_ia", "")),
        )
        return (
            cached_structured,
            f"cache:{cached_model}",
            retrieved_chunks,
        )

    # ── Step 4: Structured JSON verdict ──────────────────────────────────────
    # Gets verdict, source_law, source_url, highlights from the LLM.
    structured, used_chunks = _invoke_structured_result(prompt, retrieved_chunks)

    # ── Step 5: HYDE summary ─────────────────────────────────────────────────
    # Replaces summary_ia with a grounded, plain-language explanation built
    # exclusively from the retrieved chunks (mirrors hyde_pipeline del notebook).
    overview_generated = False
    if used_chunks:
        context_block = _build_context_block(used_chunks)

        # For overview queries, prepend the pre-computed catalog summary as
        # high-priority context so the LLM has a reliable baseline to work from.
        if is_overview_query and law_id_hints:
            catalog_summaries: list[str] = []
            for hint_id in law_id_hints:
                entry = rag_store.get_law_catalog_entry(hint_id)
                if entry and entry.get("summary_text"):
                    catalog_summaries.append(
                        f"[Resumen oficial Ley {hint_id}]\n{entry['summary_text']}"
                    )
            if catalog_summaries:
                context_block = "\n\n".join(catalog_summaries) + "\n\n" + context_block

        hyde_summary = _generate_hyde_summary(
            prompt,
            context_block,
            str(structured.get("verdict", "SIN DATOS SUFICIENTES")),
        )
        if hyde_summary:
            structured["summary_ia"] = hyde_summary

        if is_overview_query:
            overview_summary = _generate_law_overview_summary(prompt, context_block)
            if overview_summary:
                structured["summary_ia"] = overview_summary
                overview_generated = True

    # Overview queries are descriptive, not truth-claims. Keep a neutral verdict
    # so the UI doesn't render them as VERDADERO/FALSO.
    if is_overview_query:
        structured["verdict"] = "SIN DATOS SUFICIENTES"

    # Normalization enforces verdict-to-summary consistency.
    # Skip for overview queries that already produced a verified plain-language answer.
    if not overview_generated:
        structured["summary_ia"] = _normalize_summary_for_verdict(
            str(structured.get("verdict", "SIN DATOS SUFICIENTES")),
            str(structured.get("summary_ia", "")),
        )

    source = settings.OPENAI_MODEL if settings.OPENAI_API_KEY else "mock"
    _cache_structured_answer(prompt, structured, source, law_ids=law_id_hints)
    return structured, source, used_chunks
