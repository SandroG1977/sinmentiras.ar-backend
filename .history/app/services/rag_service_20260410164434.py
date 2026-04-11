import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.core.config import settings


@dataclass
class RAGChunk:
    text: str
    source: str


class LocalRAGService:
    def __init__(self) -> None:
        self._chunks: list[RAGChunk] = []
        self._loaded = False
        self._lock = Lock()

    def _iter_knowledge_files(self) -> list[Path]:
        files: list[Path] = []
        for raw_path in settings.RAG_KNOWLEDGE_PATHS:
            path = Path(raw_path)
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json"}:
                files.append(path)
                continue

            if not path.is_dir():
                continue

            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in {
                    ".md",
                    ".txt",
                    ".json",
                }:
                    files.append(file_path)

        # Keep deterministic order for stable retrieval in tests and local runs.
        return sorted(files, key=lambda p: str(p))

    def _split_text(self, text: str, max_chars: int = 700) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [cleaned]

        chunks: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= max_chars:
                chunks.append(paragraph)
                continue

            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                chunks.append(paragraph[start:end].strip())
                start = end

        return [chunk for chunk in chunks if chunk]

    def _json_to_text(self, node: object) -> str:
        if isinstance(node, dict):
            return " ".join(self._json_to_text(v) for v in node.values())
        if isinstance(node, list):
            return " ".join(self._json_to_text(item) for item in node)
        if isinstance(node, (str, int, float, bool)):
            return str(node)
        return ""

    def _read_file_text(self, file_path: Path) -> str:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        if file_path.suffix.lower() != ".json":
            return content

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return content

        return self._json_to_text(parsed)

    def _load_chunks(self) -> None:
        with self._lock:
            if self._loaded:
                return

            chunks: list[RAGChunk] = []
            for file_path in self._iter_knowledge_files():
                text = self._read_file_text(file_path)
                for chunk in self._split_text(text):
                    chunks.append(RAGChunk(text=chunk, source=str(file_path)))

            self._chunks = chunks
            self._loaded = True

    def refresh(self) -> None:
        with self._lock:
            self._loaded = False
            self._chunks = []

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z0-9_]{2,}", text.lower())}

    def retrieve(self, query: str, top_k: int) -> list[dict[str, str | float]]:
        if not settings.RAG_ENABLED:
            return []

        self._load_chunks()
        if not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored_chunks: list[tuple[float, RAGChunk]] = []
        for chunk in self._chunks:
            chunk_tokens = self._tokenize(chunk.text)
            if not chunk_tokens:
                continue

            overlap = len(query_tokens.intersection(chunk_tokens))
            if overlap == 0:
                continue

            score = overlap / math.sqrt(len(chunk_tokens))
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        selected = scored_chunks[: max(top_k, 0)]

        return [
            {
                "text": chunk.text,
                "source": chunk.source,
                "score": round(score, 4),
            }
            for score, chunk in selected
        ]


rag_service = LocalRAGService()
