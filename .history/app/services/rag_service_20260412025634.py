import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import cast
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup  # type: ignore[reportMissingImports]

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
    _ARTICLE_SUFFIXES = {
        "bis",
        "ter",
        "quater",
        "quinquies",
        "sexies",
        "septies",
        "octies",
        "nonies",
    }

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

            for row in rag_store.list_chunks():
                text = str(row.get("text", ""))
                source = str(row.get("source", "db"))
                metadata = row.get("metadata")
                if text:
                    chunks.append(
                        RAGChunk(
                            text=text,
                            source=source,
                            metadata=metadata if isinstance(metadata, dict) else {},
                        )
                    )

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
                    import faiss  # type: ignore[reportMissingImports]
                    import numpy as np

                    matrix = np.array(self._vectors, dtype="float32")
                    index = faiss.IndexFlatIP(matrix.shape[1])
                    index.add(matrix)
                    self._faiss_index = index
                except Exception:
                    self._faiss_index = None

            self._loaded = True

    def refresh(self) -> None:
        with self._lock:
            self._loaded = False
            self._chunks = []
            self._vectors = []
            self._faiss_index = None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z0-9_]{2,}", text.lower())}

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
            r'<a href="/infolegInternet/verNorma\\.do(?:;[^?]*)?\\?id=(\\d+)"',
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

    def _url_norma_fallback(self, infoleg_id: int) -> str:
        low = (infoleg_id // 10000) * 10000
        high = low + 9999
        return f"{self._INFOLEG_ATTACH_BASE}/{low}-{high}/{infoleg_id}/norma.htm"

    def _url_vinculos(self, infoleg_id: int, modo: int) -> str:
        return f"{self._INFOLEG_BASE}/verVinculos.do?modo={modo}&id={infoleg_id}"

    def _extraer_url_texact_desde_ver_norma(
        self,
        ver_norma_html: str,
        ver_norma_url: str,
    ) -> str | None:
        soup = BeautifulSoup(ver_norma_html, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "texact.htm" in href.lower():
                return urljoin(ver_norma_url, href)
        return None

    def _heading_level(self, title: str) -> int:
        text = title.upper()
        if re.search(r"\b(LIBRO|ANEXO)\b", text):
            return 0
        if re.search(r"T[ÍI]TULO", text):
            return 1
        if re.search(r"CAP[ÍI]TULO", text):
            return 2
        if re.search(r"SECCI[ÓO]N", text):
            return 3
        return 99

    def _get_capitulo(self, stack: list[str]) -> str | None:
        for entry in reversed(stack):
            if re.search(r"CAP[ÍI]TULO", entry.upper()):
                return entry
        return None

    def _parse_articulo_ref(
        self,
        texto_art: str,
    ) -> tuple[int | None, str | None, str | None]:
        match = re.search(
            r"\b(?:Art\.|Artículo)\s*(\d+)\s*([a-zA-Z]+)?",
            texto_art,
            flags=re.IGNORECASE,
        )
        if not match:
            return None, None, None
        nro = int(match.group(1))
        suf = match.group(2).lower() if match.group(2) else None
        if suf and suf not in self._ARTICLE_SUFFIXES:
            suf = None
        ref = f"{nro} {suf}".strip() if suf else str(nro)
        return nro, suf, ref

    def _es_nota_substitucion_texto(self, texto: str) -> bool:
        body = texto.lower()
        patrones = [
            "nota infoleg",
            "sustituido por art.",
            "sustituida por art.",
            "incorporado por art.",
            "incorporada por art.",
            "modificado por art.",
            "modificada por art.",
            "texto segun",
            "vigencia:",
        ]
        return any(patron in body for patron in patrones)

    def _es_nota_substitucion(self, block: str) -> bool:
        return self._es_nota_substitucion_texto(block) and len(block) < 1500

    def _extraer_notas_finales(self, block: str) -> tuple[str, list[str]]:
        notas: list[str] = []
        texto = block.strip()

        while True:
            match = re.search(r"\s*(\([^()]{20,1200}\))\s*$", texto, re.IGNORECASE)
            if not match:
                break
            candidata = match.group(1)
            if not self._es_nota_substitucion_texto(candidata):
                break
            notas.insert(0, candidata)
            texto = texto[: match.start()].rstrip()

        return texto, notas

    def _split_ley_semantico_articulos(
        self,
        texto: str,
    ) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
        heading_start_re = re.compile(
            r"\*{0,2}\s*(?:T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|LIBRO|ANEXO)\s+[IVXLCDM0-9]"
        )
        article_header_pattern = (
            r"\*{0,2}\s*(?:Art\.|Artículo)\s*\d+[a-zA-Zº°]*\.?\s*"
            r"(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies)?\s*[—-]\s*"
        )
        article_start_re = re.compile(article_header_pattern, re.IGNORECASE)
        marker_re = re.compile(
            r"\*{0,2}\s*(?:T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|LIBRO|ANEXO)\s+[IVXLCDM0-9]+|"
            + article_header_pattern,
            re.IGNORECASE,
        )

        markers = list(marker_re.finditer(texto))
        if not markers:
            return (
                [
                    {
                        "kind": "texto",
                        "title": "TEXTO",
                        "text": texto,
                        "path": "",
                        "articulo_nro": None,
                        "articulo_suffix": None,
                        "articulo_ref": None,
                        "capitulo": None,
                    }
                ],
                {},
            )

        chunks: list[dict[str, object]] = []
        notes_by_ref: dict[str, list[str]] = {}
        heading_stack: list[str] = []

        if markers[0].start() > 0:
            pre = texto[: markers[0].start()].strip()
            if pre:
                chunks.append(
                    {
                        "kind": "preambulo",
                        "title": "PREAMBULO",
                        "text": pre,
                        "path": "",
                        "articulo_nro": None,
                        "articulo_suffix": None,
                        "articulo_ref": None,
                        "capitulo": None,
                    }
                )

        for index, marker in enumerate(markers):
            start = marker.start()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(texto)
            block = texto[start:end].strip()
            if not block:
                continue

            if heading_start_re.match(block):
                title = re.split(
                    r"(?=" + article_header_pattern + r")",
                    block,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip()

                new_level = self._heading_level(title)
                while (
                    heading_stack
                    and self._heading_level(heading_stack[-1]) >= new_level
                ):
                    heading_stack.pop()
                heading_stack.append(title)

                chunks.append(
                    {
                        "kind": "heading",
                        "title": title,
                        "text": block,
                        "path": " > ".join(heading_stack),
                        "articulo_nro": None,
                        "articulo_suffix": None,
                        "articulo_ref": None,
                        "capitulo": self._get_capitulo(heading_stack),
                    }
                )
                continue

            if article_start_re.match(block):
                articulo_nro, articulo_suffix, articulo_ref = self._parse_articulo_ref(
                    marker.group()
                )

                if self._es_nota_substitucion(block):
                    ref = articulo_ref or "sin_ref"
                    notes_by_ref.setdefault(ref, []).append(block)
                    continue

                block_limpio, notas_finales = self._extraer_notas_finales(block)
                if notas_finales:
                    ref = articulo_ref or "sin_ref"
                    notes_by_ref.setdefault(ref, []).extend(notas_finales)
                block = block_limpio

                match_title = re.match(
                    r"(?s)(" + article_header_pattern + r"[^*\n]{0,180})",
                    block,
                    flags=re.IGNORECASE,
                )
                title = match_title.group(1).strip() if match_title else block[:180]
                chunks.append(
                    {
                        "kind": "articulo",
                        "title": title,
                        "text": block,
                        "path": " > ".join(heading_stack),
                        "articulo_nro": articulo_nro,
                        "articulo_suffix": articulo_suffix,
                        "articulo_ref": articulo_ref,
                        "capitulo": self._get_capitulo(heading_stack),
                    }
                )
                continue

            chunks.append(
                {
                    "kind": "texto",
                    "title": block[:180],
                    "text": block,
                    "path": " > ".join(heading_stack),
                    "articulo_nro": None,
                    "articulo_suffix": None,
                    "articulo_ref": None,
                    "capitulo": self._get_capitulo(heading_stack),
                }
            )

        return chunks, notes_by_ref

    def _fetch_infoleg_law(self, law_id: int) -> tuple[str, dict[str, object]]:
        infoleg_id, search_url = self._buscar_id_norma_infoleg(law_id)
        ver_norma_url = self._url_ver_norma(infoleg_id)

        ver_response = requests.get(ver_norma_url, timeout=20)
        ver_response.raise_for_status()
        ver_response.encoding = ver_response.apparent_encoding or "utf-8"

        url_texact = self._extraer_url_texact_desde_ver_norma(
            ver_response.text,
            ver_norma_url,
        )
        if not url_texact:
            url_texact = self._url_texact_fallback(infoleg_id)

        if "texact.htm" in url_texact.lower():
            url_norma = re.sub(
                r"texact\.htm(?:\?.*)?$",
                "norma.htm",
                url_texact,
                flags=re.IGNORECASE,
            )
        else:
            url_norma = self._url_norma_fallback(infoleg_id)

        texact_response = requests.get(url_texact, timeout=30)
        texact_response.raise_for_status()
        texact_response.encoding = texact_response.apparent_encoding or "utf-8"

        sha256_hash = hashlib.sha256(texact_response.content).hexdigest()
        soup = BeautifulSoup(texact_response.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        title_text = soup.title.get_text(strip=True) if soup.title else f"Ley {law_id}"

        metadata = {
            "source": "INFOLEG",
            "status": "VIGENTE",
            "law_id": law_id,
            "infoleg_id": infoleg_id,
            "source_norma": url_norma,
            "source_actualizado": url_texact,
            "source_url": url_texact,
            "search_url": search_url,
            "ver_norma_url": ver_norma_url,
            "vinculos_actualiza_url": self._url_vinculos(infoleg_id, modo=1),
            "vinculos_actualizado_por_url": self._url_vinculos(infoleg_id, modo=2),
            "sha256_hash": sha256_hash,
            "titulo": title_text,
        }
        return text, metadata

    def _build_infoleg_chunk_records(
        self,
        text: str,
    ) -> list[dict[str, object]]:
        normalized_text = re.sub(r"\s+", " ", text).strip()
        chunks_estructurados, notes_by_ref = self._split_ley_semantico_articulos(
            normalized_text
        )

        chunk_records: list[dict[str, object]] = []
        chunk_count = len(chunks_estructurados)
        for index, chunk in enumerate(chunks_estructurados, start=1):
            ref = str(chunk.get("articulo_ref") or "") or None
            notas_substitucion = notes_by_ref.get(ref, []) if ref else []
            chunk_records.append(
                {
                    "text": str(chunk.get("text", "")).strip(),
                    "metadata": {
                        "chunk_index": index,
                        "chunk_count": chunk_count,
                        "kind": chunk.get("kind"),
                        "title": chunk.get("title"),
                        "path": chunk.get("path"),
                        "articulo_nro": chunk.get("articulo_nro"),
                        "articulo_suffix": chunk.get("articulo_suffix"),
                        "articulo_ref": chunk.get("articulo_ref"),
                        "capitulo": chunk.get("capitulo"),
                        "notas_substitucion": notas_substitucion,
                        "cantidad_notas_substitucion": len(notas_substitucion),
                    },
                }
            )
        return chunk_records

    def ingest_law_from_infoleg(
        self,
        law_id: int,
        *,
        replace_existing: bool = True,
    ) -> InfolegIngestResult:
        text, metadata = self._fetch_infoleg_law(law_id)
        chunk_records = self._build_infoleg_chunk_records(text)
        if not chunk_records:
            raise ValueError(f"No se pudieron generar chunks para la ley {law_id}")

        infoleg_id = int(cast(int | str, metadata["infoleg_id"]))
        source_uri = str(cast(str, metadata["source_actualizado"]))
        document_id = hashlib.sha1(
            f"infoleg:{law_id}:{infoleg_id}".encode("utf-8")
        ).hexdigest()

        inserted = rag_store.ingest_chunk_records(
            document_id=document_id,
            source_uri=source_uri,
            chunk_records=chunk_records,
            base_metadata=cast(
                dict[str, str | int | float | bool | list[str] | None],
                metadata,
            ),
            replace_existing=replace_existing,
        )
        self.refresh()

        return InfolegIngestResult(
            document_id=document_id,
            source_uri=source_uri,
            law_id=law_id,
            infoleg_id=infoleg_id,
            chunks_inserted=inserted,
            source_norma=str(metadata["source_norma"]),
            source_actualizado=source_uri,
            sha256_hash=str(metadata["sha256_hash"]),
        )

    def retrieve(self, query: str, top_k: int) -> list[dict[str, object]]:
        if not settings.RAG_ENABLED:
            return []

        self._load_chunks()
        if not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

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
