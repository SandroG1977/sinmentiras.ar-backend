import hashlib
import html
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from urllib.parse import urljoin

import requests

from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.rag_store import rag_store


@dataclass
class RAGChunk:
    text: str
    source: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class InfolegIngestResult:
    document_id: str
    source_uri: str
    law_id: int
    infoleg_id: int
    chunks_inserted: int
    source_norma: str
    source_actualizado: str
    sha256_hash: str


class LocalRAGService:
    _INFOLEG_BASE = "https://servicios.infoleg.gob.ar/infolegInternet"
    _INFOLEG_ATTACH_BASE = f"{_INFOLEG_BASE}/anexos"

    def __init__(self) -> None:
        self._chunks: list[RAGChunk] = []
        self._vectors: list[list[float]] = []
        self._faiss_index = None
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

            # DB-backed chunks from ingested documents (e.g., MinIO files).
            for row in rag_store.list_chunks():
                text = row.get("text", "")
                source = row.get("source", "db")
                if text:
                    chunks.append(RAGChunk(text=text, source=source))

            # Local file-based fallback knowledge.
            for file_path in self._iter_knowledge_files():
                text = self._read_file_text(file_path)
                for chunk in self._split_text(text):
                    chunks.append(RAGChunk(text=chunk, source=str(file_path)))

            self._chunks = chunks
            self._vectors = (
                embedding_service.embed_texts([chunk.text for chunk in chunks])
                if chunks
                else []
            )

            self._faiss_index = None
            if settings.RAG_USE_FAISS and self._vectors:
                try:
                    import numpy as np
                    import faiss

                    matrix = np.array(self._vectors, dtype="float32")
                    index = faiss.IndexFlatIP(matrix.shape[1])
                    index.add(matrix)
                    self._faiss_index = index
                except Exception:
                    # Fallback to lexical scoring when faiss is unavailable.
                    self._faiss_index = None

            self._loaded = True

    def refresh(self) -> None:
        with self._lock:
            self._loaded = False
            self._chunks = []
            self._vectors = []
            self._faiss_index = None

    def _buscar_id_norma_infoleg(
        self,
        numero_ley: int,
        tipo_norma: int = 1,
        anio_sancion: str = "",
    ) -> tuple[int, str]:
        search_url = (
            f"{self._INFOLEG_BASE}/buscarNormas.do?tipoNorma={tipo_norma}"
            f"&numero={numero_ley}&anioSancion={anio_sancion}"
        )
        response = requests.get(search_url, timeout=20)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"

        match = re.search(
            r'<a\s+href="[^"]*verNorma\.do[^"]*?\bid=(\d+)',
            response.text,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                f"No se pudo extraer idNorma para la ley {numero_ley} desde Infoleg"
            )

        return int(match.group(1)), search_url

    def _url_ver_norma(self, infoleg_id: int) -> str:
        return f"{self._INFOLEG_BASE}/verNorma.do?id={infoleg_id}"

    def _url_texact_fallback(self, infoleg_id: int) -> str:
        low = (infoleg_id // 10000) * 10000
        high = low + 9999
        return f"{self._INFOLEG_ATTACH_BASE}/{low}-{high}/{infoleg_id}/texact.htm"

    def _url_norma_from_texact(self, texact_url: str, infoleg_id: int) -> str:
        if "texact.htm" in texact_url.lower():
            return re.sub(
                r"texact\.htm(?:\?.*)?$",
                "norma.htm",
                texact_url,
                flags=re.IGNORECASE,
            )
        low = (infoleg_id // 10000) * 10000
        high = low + 9999
        return f"{self._INFOLEG_ATTACH_BASE}/{low}-{high}/{infoleg_id}/norma.htm"

    def _extraer_url_texact_desde_ver_norma(
        self,
        ver_norma_html: str,
        ver_norma_url: str,
    ) -> str | None:
        match = re.search(
            r'href="([^"]*texact\.htm[^"]*)"',
            ver_norma_html,
            re.IGNORECASE,
        )
        if not match:
            return None
        return urljoin(ver_norma_url, match.group(1))

    def _html_to_text(self, raw_html: str) -> str:
        no_script = re.sub(
            r"<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>",
            " ",
            raw_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"<[^>]+>", " ", no_script)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def ingest_law_from_infoleg(
        self,
        law_id: int,
        *,
        replace_existing: bool = True,
    ) -> InfolegIngestResult:
        infoleg_id, _search_url = self._buscar_id_norma_infoleg(law_id)
        ver_norma_url = self._url_ver_norma(infoleg_id)

        ver_response = requests.get(ver_norma_url, timeout=20)
        ver_response.raise_for_status()
        ver_response.encoding = ver_response.apparent_encoding or "utf-8"

        texact_url = self._extraer_url_texact_desde_ver_norma(
            ver_response.text,
            ver_norma_url,
        ) or self._url_texact_fallback(infoleg_id)

        texact_response = requests.get(texact_url, timeout=30)
        texact_response.raise_for_status()
        texact_response.encoding = texact_response.apparent_encoding or "utf-8"

        sha256_hash = hashlib.sha256(texact_response.content).hexdigest()
        normalized_text = self._html_to_text(texact_response.text)
        chunks = self._split_text(normalized_text)

        source_norma = self._url_norma_from_texact(texact_url, infoleg_id)
        source_uri = texact_url
        document_id = f"infoleg-{infoleg_id}-{sha256_hash[:12]}"

        inserted = rag_store.ingest_document(
            document_id=document_id,
            source_uri=source_uri,
            chunks=chunks,
            metadata={
                "source": "INFOLEG",
                "status": "VIGENTE",
                "law_id": law_id,
                "infoleg_id": infoleg_id,
                "source_norma": source_norma,
                "source_actualizado": texact_url,
                "source_url": texact_url,
                "sha256_hash": sha256_hash,
            },
            replace_existing=replace_existing,
        )

        self.refresh()

        return InfolegIngestResult(
            document_id=document_id,
            source_uri=source_uri,
            law_id=law_id,
            infoleg_id=infoleg_id,
            chunks_inserted=inserted,
            source_norma=source_norma,
            source_actualizado=texact_url,
            sha256_hash=sha256_hash,
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z0-9_]{2,}", text.lower())}

    def retrieve(self, query: str, top_k: int) -> list[dict[str, object]]:
        if not settings.RAG_ENABLED:
            return []

        self._load_chunks()
        if not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Preferred path: semantic retrieval via FAISS.
        if self._faiss_index is not None and self._vectors:
            try:
                import numpy as np

                query_vector = embedding_service.embed_texts([query])[0]
                matrix = np.array([query_vector], dtype="float32")
                scores, indexes = self._faiss_index.search(matrix, max(top_k, 0))

                results: list[dict[str, object]] = []
                for score, idx in zip(scores[0], indexes[0]):
                    if idx < 0 or idx >= len(self._chunks):
                        continue
                    chunk = self._chunks[idx]
                    results.append(
                        {
                            "text": chunk.text,
                            "source": chunk.source,
                            "score": round(float(score), 4),
                            "metadata": chunk.metadata,
                        }
                    )
                if results:
                    return results
            except Exception:
                pass

        # Fallback: lexical token overlap.
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
                "metadata": chunk.metadata,
            }
            for score, chunk in selected
        ]


rag_service = LocalRAGService()
