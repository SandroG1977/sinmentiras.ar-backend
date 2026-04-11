from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    output: str
    used_model: str
    source: str = "langgraph"
