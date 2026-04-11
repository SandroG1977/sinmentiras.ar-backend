from pydantic import BaseModel, Field


class NewsContext(BaseModel):
    source: str
    title: str
    sentiment: str  # "positive" | "negative" | "neutral"


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)


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
    used_model: str
