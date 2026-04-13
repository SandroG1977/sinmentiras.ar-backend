import dotenv
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.services.text_chunker import chunk_text
from app.services.question_cache_store import question_cache_store
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
            temperature=0,
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


def _extract_law_id_hint(prompt: str) -> int | None:
    match = re.search(
        r"\bley\s*(?:n(?:ro)?\.?|n°)?\s*(\d{1,3}(?:[\.,]\d{3})+|\d{4,6})\b",
        prompt,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    normalized = re.sub(r"[\.,]", "", match.group(1))
    try:
        law_id = int(normalized)
    except ValueError:
        return None

    return law_id if law_id > 0 else None


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
    law_id_hint: int | None,
    top_k: int,
) -> list[dict[str, object]]:
    if law_id_hint is None:
        return chunks

    matching = [chunk for chunk in chunks if _metadata_law_id(chunk) == law_id_hint]
    non_matching = [chunk for chunk in chunks if _metadata_law_id(chunk) != law_id_hint]
    if matching:
        return (matching + non_matching)[:top_k]
    return chunks


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


def _cache_structured_answer(prompt: str, structured: dict, source: str) -> None:
    question_cache_store.save_answer(
        question=prompt,
        answer=json.dumps(structured, ensure_ascii=False),
        used_model=source,
    )


def run_agent(prompt: str) -> tuple[dict, str, list[dict[str, object]]]:
    law_id_hint = _extract_law_id_hint(prompt)
    retrieved_chunks = (
        _prioritize_chunks_for_law(
            rag_service.retrieve(prompt, top_k=settings.RAG_TOP_K * 3),
            law_id_hint,
            settings.RAG_TOP_K,
        )
        if settings.RAG_ENABLED
        else []
    )

    query_chunks = _build_query_chunks(prompt)
    best_cached = None if law_id_hint is not None else _find_best_cached(query_chunks)

    if best_cached:
        cached_answer_str, cached_model, _similarity = best_cached
        return (
            _parse_cached_answer(cached_answer_str),
            f"cache:{cached_model}",
            retrieved_chunks,
        )

    structured, used_chunks = _invoke_structured_result(prompt, retrieved_chunks)
    source = settings.OPENAI_MODEL if settings.OPENAI_API_KEY else "mock"
    _cache_structured_answer(prompt, structured, source)
    return structured, source, used_chunks
