import hashlib
import math
import dotenv

from app.core.config import settings

doenv.load_dotenv()  # Load environment variables from .env file, if present


class EmbeddingService:
    def __init__(self, dim: int = 256) -> None:
        self._fallback_dim = dim

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _fallback_embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._fallback_dim
        tokens = re.findall(r"[A-Za-z0-9_]{2,}", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._fallback_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign

        return self._normalize(vector)

    def _fallback_embed(self, texts: list[str]) -> list[list[float]]:
        return [self._fallback_embed_one(text) for text in texts]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if not settings.OPENAI_API_KEY:
            return self._fallback_embed(texts)

        try:
            from langchain_openai import OpenAIEmbeddings

            embedding_client = OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )
            vectors = embedding_client.embed_documents(texts)
            return [
                self._normalize([float(value) for value in vector])
                for vector in vectors
            ]
        except Exception:
            # Fallback keeps the app functional if remote embedding is unavailable.
            return self._fallback_embed(texts)


embedding_service = EmbeddingService()
