from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse, NewsContext
from app.services.agent_graph import make_audit_id, make_response_hash, run_agent

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    result, used_model = run_agent(payload.prompt)

    news_context = [
        NewsContext(
            source=item.get("source", ""),
            title=item.get("title", ""),
            sentiment=item.get("sentiment", "neutral"),
        )
        for item in (result.get("news_context") or [])
        if isinstance(item, dict)
    ]

    summary = result.get("summary_ia", "")
    return ChatResponse(
        id=make_audit_id(),
        query=payload.prompt,
        verdict=result.get("verdict", "SIN DATOS SUFICIENTES"),
        summary_ia=summary,
        hash=make_response_hash(payload.prompt, summary),
        source_law=result.get("source_law", ""),
        source_url=result.get("source_url", ""),
        original_text=result.get("original_text", ""),
        highlights=result.get("highlights") or [],
        news_context=news_context,
        used_model=used_model,
    )
