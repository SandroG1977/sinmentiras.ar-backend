from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent_graph import run_agent

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    output, used_model = run_agent(payload.prompt)
    return ChatResponse(output=output, used_model=used_model)
