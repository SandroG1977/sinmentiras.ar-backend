from fastapi import APIRouter

from app.api.v1.endpoints import chat, health, rag

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, prefix="/agent", tags=["agent"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
