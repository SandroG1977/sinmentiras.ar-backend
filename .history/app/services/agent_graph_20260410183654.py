import dotenv
import hashlib
import json
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

_SYSTEM_PROMPT = (
    "Eres un verificador legal para legislación argentina. "
    "Analiza el contexto proporcionado de documentos legales oficiales y responde "
    "ÚNICAMENTE con un objeto JSON válido (sin markdown, sin bloques de código) "
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
    "No incluyas ningún texto fuera del JSON."
)


class AgentState(TypedDict):
    prompt: str
    output: str
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

        if settings.RAG_ENABLED:
            retrieved = rag_service.retrieve(prompt, top_k=settings.RAG_TOP_K)
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


def run_agent(prompt: str) -> tuple[dict, str]:
    query_chunks = [prompt]
    if settings.CACHE_QUERY_CHUNKING_ENABLED:
        chunked = chunk_text(
            prompt,
            chunk_size=settings.CACHE_QUERY_CHUNK_SIZE,
            chunk_overlap=settings.CACHE_QUERY_CHUNK_OVERLAP,
        )
        query_chunks.extend([chunk for chunk in chunked if chunk and chunk != prompt])

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

    if best_cached:
        cached_answer_str, cached_model, _similarity = best_cached
        try:
            cached_result = json.loads(cached_answer_str)
        except (json.JSONDecodeError, ValueError):
            cached_result = _make_fallback_result(cached_answer_str)
        return cached_result, f"cache:{cached_model}"

    result = agent_graph.invoke(
        {"prompt": prompt, "output": "", "context_chunks": [], "structured_result": {}}
    )
    structured = result.get("structured_result") or _make_fallback_result(
        result.get("output", "")
    )
    source = settings.OPENAI_MODEL if settings.OPENAI_API_KEY else "mock"
    question_cache_store.save_answer(
        question=prompt,
        answer=json.dumps(structured, ensure_ascii=False),
        used_model=source,
    )
    return structured, source
