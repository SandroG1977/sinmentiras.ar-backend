import dotenv
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.services.text_chunker import chunk_text
from app.services.question_cache_store import question_cache_store
from app.services.rag_service import rag_service


dotenv.load_dotenv()  # Load environment variables from .env file, if present


class AgentState(TypedDict):
    prompt: str
    output: str
    context_chunks: list[str]


def _build_graph():
    def run_model(state: AgentState) -> AgentState:
        prompt = state["prompt"]
        context_chunks: list[str] = []
        prompt_with_context = prompt

        if settings.RAG_ENABLED:
            retrieved = rag_service.retrieve(prompt, top_k=settings.RAG_TOP_K)
            if retrieved:
                context_chunks = [
                    f"[{idx}] Source: {item['source']}\n{item['text']}"
                    for idx, item in enumerate(retrieved, start=1)
                ]
                context_block = "\n\n".join(context_chunks)
                prompt_with_context = (
                    "Use the context snippets below as the primary source of truth. "
                    "If context is insufficient, say what is missing and answer conservatively.\n\n"
                    f"Context:\n{context_block}\n\n"
                    f"User question:\n{prompt}"
                )

        # Fallback to a deterministic mock response when no API key is set.
        if not settings.OPENAI_API_KEY:
            return {
                "prompt": prompt,
                "output": f"[mock-response] {prompt}",
                "context_chunks": context_chunks,
            }

        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=lambda: settings.OPENAI_API_KEY or "",
            temperature=0,
        )
        response = llm.invoke(prompt_with_context)
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return {
            "prompt": prompt,
            "output": content,
            "context_chunks": context_chunks,
        }

    graph = StateGraph(AgentState)
    graph.add_node("run_model", run_model)
    graph.add_edge(START, "run_model")
    graph.add_edge("run_model", END)
    return graph.compile()


agent_graph = _build_graph()


def run_agent(prompt: str) -> tuple[str, str]:
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
        output, cached_model, _similarity = best_cached
        return output, f"cache:{cached_model}"

    result = agent_graph.invoke({"prompt": prompt, "output": "", "context_chunks": []})
    source = settings.OPENAI_MODEL if settings.OPENAI_API_KEY else "mock"
    question_cache_store.save_answer(
        question=prompt,
        answer=result["output"],
        used_model=source,
    )
    return result["output"], source
