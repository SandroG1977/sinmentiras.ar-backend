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
from bs4 import BeautifulSoup

from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.rag_store import rag_store


@dataclass
class RAGChunk:
    """Unidad recuperable del índice RAG con su metadata asociada."""

    text: str
    source: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class InfolegIngestResult:
    """Resultado resumido de una ingesta de ley desde InfoLEG."""

    document_id: str
    source_uri: str
    law_id: int
    infoleg_id: int
    chunks_inserted: int
    source_norma: str
    source_actualizado: str
    sha256_hash: str


class LocalRAGService:
    """Servicio RAG local para cargar, enriquecer, persistir y recuperar contexto legal."""

    _INFOLEG_BASE = "https://servicios.infoleg.gob.ar/infolegInternet"
    _INFOLEG_ATTACH_BASE = f"{_INFOLEG_BASE}/anexos"

    def __init__(self) -> None:
        """Inicializa caches en memoria y sincronización para carga diferida."""

        self._chunks: list[RAGChunk] = []
        self._vectors: list[list[float]] = []
        self._faiss_index = None
        self._loaded = False
        self._lock = Lock()

    # Descubrimiento y carga de conocimiento local.
    def _iter_knowledge_files(self) -> list[Path]:
        """Devuelve los archivos fuente permitidos para conocimiento local."""

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
        """Parte texto libre en chunks simples cuando no hay estructura semántica."""

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
        """Aplana estructuras JSON anidadas a texto plano indexable."""

        if isinstance(node, dict):
            return " ".join(self._json_to_text(v) for v in node.values())
        if isinstance(node, list):
            return " ".join(self._json_to_text(item) for item in node)
        if isinstance(node, (str, int, float, bool)):
            return str(node)
        return ""

    def _read_file_text(self, file_path: Path) -> str:
        """Lee archivos locales y convierte JSON a texto antes de indexar."""

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        if file_path.suffix.lower() != ".json":
            return content

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return content

        return self._json_to_text(parsed)

    def _load_chunks(self) -> None:
        """Carga chunks desde la base y el conocimiento local, y construye embeddings."""

        with self._lock:
            if self._loaded:
                return

            chunks: list[RAGChunk] = []

            # DB-backed chunks from ingested documents (e.g., MinIO files).
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
        """Invalida el estado en memoria para forzar una recarga en el próximo uso."""

        with self._lock:
            self._loaded = False
            self._chunks = []
            self._vectors = []
            self._faiss_index = None

    # Resolución y armado de URLs de InfoLEG.
    def _buscar_id_norma_infoleg(
        self,
        numero_ley: int,
        tipo_norma: int = 1,
        anio_sancion: str = "",
    ) -> tuple[int, str]:
        """Busca el id interno de InfoLEG a partir del número de ley."""

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
        """Construye la URL de verNorma para una norma de InfoLEG."""

        return f"{self._INFOLEG_BASE}/verNorma.do?id={infoleg_id}"

    def _url_texact_fallback(self, infoleg_id: int) -> str:
        """Genera una URL de texto actualizado cuando no se pudo descubrir desde HTML."""

        low = (infoleg_id // 5000) * 5000
        high = low + 9999
        return f"{self._INFOLEG_ATTACH_BASE}/{low}-{high}/{infoleg_id}/texact.htm"

    def _url_norma_fallback(self, infoleg_id: int) -> str:
        """Genera una URL de norma base cuando no se pudo inferir desde texact."""

        low = (infoleg_id // 5000) * 5000
        high = low + 9999
        return f"{self._INFOLEG_ATTACH_BASE}/{low}-{high}/{infoleg_id}/norma.htm"

    def _url_vinculos(self, infoleg_id: int, modo: int) -> str:
        """Construye las URLs de vínculos de actualización de InfoLEG."""

        return f"{self._INFOLEG_BASE}/verVinculos.do?modo={modo}&id={infoleg_id}"

    def _url_norma_from_texact(self, texact_url: str, infoleg_id: int) -> str:
        """Deriva la URL de norma a partir de texact o usa un fallback sintético."""

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
        """Extrae la URL de texto actualizado desde la página verNorma."""

        soup = BeautifulSoup(ver_norma_html, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href = str(anchor["href"])
            if "texact.htm" in href.lower():
                return urljoin(ver_norma_url, href)
        return None

    # Normalización y chunking semántico de leyes.
    def _split_ley_semantico_articulos(
        self,
        texto: str,
    ) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
        """Divide una ley en preámbulo, headings, artículos y texto libre asociado."""

        heading_start_re = re.compile(
            r"\*{0,2}\s*(?:T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|LIBRO|ANEXO)\s+[IVXLCDM0-9]+"
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

        def heading_level(title: str) -> int:
            title_upper = title.upper()
            if re.search(r"\bLIBRO\b", title_upper):
                return 0
            if re.search(r"\bANEXO\b", title_upper):
                return 0
            if re.search(r"T[ÍI]TULO", title_upper):
                return 1
            if re.search(r"CAP[ÍI]TULO", title_upper):
                return 2
            if re.search(r"SECCI[ÓO]N", title_upper):
                return 3
            return 99

        def get_capitulo(stack: list[str]) -> str | None:
            for entry in reversed(stack):
                if re.search(r"CAP[ÍI]TULO", entry.upper()):
                    return entry
            return None

        def parse_articulo_ref(
            texto_art: str,
        ) -> tuple[int | None, str | None, str | None]:
            match = re.search(
                r"\b(?:Art\.|Artículo)\s*(\d+)\s*([a-zA-Z]+)?",
                texto_art,
                flags=re.IGNORECASE,
            )
            if not match:
                return None, None, None
            numero = int(match.group(1))
            suffix = match.group(2).lower() if match.group(2) else None
            if suffix and suffix not in {
                "bis",
                "ter",
                "quater",
                "quinquies",
                "sexies",
                "septies",
                "octies",
                "nonies",
            }:
                suffix = None
            ref = f"{numero} {suffix}".strip() if suffix else str(numero)
            return numero, suffix, ref

        def es_nota_substitucion_texto(texto_nota: str) -> bool:
            base = texto_nota.lower()
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
            return any(pattern in base for pattern in patrones)

        def es_nota_substitucion(block: str) -> bool:
            return es_nota_substitucion_texto(block) and len(block) < 1500

        def extraer_notas_finales(block: str) -> tuple[str, list[str]]:
            notas: list[str] = []
            texto_local = block.strip()
            while True:
                match = re.search(
                    r"\s*(\([^()]{20,1200}\))\s*$",
                    texto_local,
                    flags=re.IGNORECASE,
                )
                if not match:
                    break
                candidata = match.group(1)
                if not es_nota_substitucion_texto(candidata):
                    break
                notas.insert(0, candidata)
                texto_local = texto_local[: match.start()].rstrip()
            return texto_local, notas

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
            preambulo = texto[: markers[0].start()].strip()
            if preambulo:
                chunks.append(
                    {
                        "kind": "preambulo",
                        "title": "PREAMBULO",
                        "text": preambulo,
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
                new_level = heading_level(title)
                while heading_stack and heading_level(heading_stack[-1]) >= new_level:
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
                        "capitulo": get_capitulo(heading_stack),
                    }
                )
                continue

            if article_start_re.match(block):
                articulo_nro, articulo_suffix, articulo_ref = parse_articulo_ref(
                    marker.group()
                )

                if es_nota_substitucion(block):
                    ref = articulo_ref or "sin_ref"
                    notes_by_ref.setdefault(ref, []).append(block)
                    continue

                block_limpio, notas_finales = extraer_notas_finales(block)
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
                        "capitulo": get_capitulo(heading_stack),
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
                    "capitulo": get_capitulo(heading_stack),
                }
            )

        return chunks, notes_by_ref

    def _is_infoleg_error_page(self, texto: str) -> bool:
        """Detecta respuestas HTML de error de InfoLEG disfrazadas de contenido válido."""

        normalized = re.sub(r"\s+", " ", texto).strip().lower()
        markers = [
            "no se pudo acceder al archivo solicitado",
            "error: archivo no encontrado",
        ]
        if any(marker in normalized for marker in markers):
            return True
        return False

    def _html_to_text(self, raw_html: str) -> str:
        """Convierte HTML arbitrario a texto plano conservando solo contenido visible."""

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

    # Ingesta de normas desde InfoLEG.
    def ingest_law_from_infoleg(
        self,
        law_id: int,
        *,
        replace_existing: bool = True,
    ) -> InfolegIngestResult:
        """Ingiere una ley desde InfoLEG, la fragmenta semánticamente y la persiste."""

        infoleg_id, search_url = self._buscar_id_norma_infoleg(law_id)
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
        soup = BeautifulSoup(texact_response.text, "html.parser")
        texto_fuente_directo = soup.get_text(" ", strip=True)
        texto_fuente = re.sub(r"\s+", " ", texto_fuente_directo).strip()

        if not texto_fuente:
            raise ValueError(
                f"Infoleg devolvio contenido vacio para la ley {law_id} ({texact_url})"
            )
        if self._is_infoleg_error_page(texto_fuente):
            raise ValueError(
                f"Infoleg devolvio una pagina de error para la ley {law_id} ({texact_url})"
            )

        chunks_estructurados, notes_by_ref = self._split_ley_semantico_articulos(
            texto_fuente
        )
        if not chunks_estructurados:
            raise ValueError(
                f"No se pudieron generar chunks semanticos para la ley {law_id}"
            )

        if "texact.htm" in texact_url.lower():
            source_norma = re.sub(
                r"texact\.htm(?:\?.*)?$",
                "norma.htm",
                texact_url,
                flags=re.IGNORECASE,
            )
        else:
            source_norma = self._url_norma_fallback(infoleg_id)

        title_text = soup.title.get_text(strip=True) if soup.title else f"Ley {law_id}"

        metadata_doc: dict[str, object] = {
            "source": "INFOLEG",
            "status": "VIGENTE",
            "law_id": law_id,
            "infoleg_id": infoleg_id,
            "source_norma": source_norma,
            "source_actualizado": texact_url,
            "source_url": texact_url,
            "search_url": search_url,
            "ver_norma_url": ver_norma_url,
            "vinculos_actualiza_url": self._url_vinculos(infoleg_id, modo=1),
            "vinculos_actualizado_por_url": self._url_vinculos(infoleg_id, modo=2),
            "sha256_hash": sha256_hash,
            "titulo": title_text,
        }

        source_uri = texact_url
        document_id = f"infoleg-{infoleg_id}-{sha256_hash[:12]}"

        chunk_records: list[dict[str, object]] = []
        for index, chunk in enumerate(chunks_estructurados, start=1):
            ref = chunk.get("articulo_ref")
            notas_substitucion = notes_by_ref.get(str(ref), []) if ref else []
            chunk_records.append(
                {
                    "text": str(chunk.get("text", "")),
                    "metadata": {
                        "chunk_index": index,
                        "chunk_count": len(chunks_estructurados),
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

        inserted = rag_store.ingest_chunk_records(
            document_id=document_id,
            source_uri=source_uri,
            chunk_records=chunk_records,
            base_metadata=metadata_doc,
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

    # Recuperación y ranking de contexto.
    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokeniza texto en términos simples para el fallback léxico."""

        return {token for token in re.findall(r"[A-Za-z0-9_]{2,}", text.lower())}

    def retrieve(self, query: str, top_k: int) -> list[dict[str, object]]:
        """Recupera los chunks más relevantes usando FAISS o scoring léxico."""

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
