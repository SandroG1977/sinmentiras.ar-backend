from pydantic import BaseModel, Field


class NewsContext(BaseModel):
    source: str
    title: str
    sentiment: str  # "positive" | "negative" | "neutral"


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=5, le=20)


class UsedChunk(BaseModel):
    text: str
    source: str
    score: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    id: str
    query: str
    verdict: str
    summary_ia: str
    hash: str
    source_law: str
    source_url: str
    original_text: str
    highlights: list[str]
    news_context: list[NewsContext]
    chunks_used: list[UsedChunk]
    used_model: str
