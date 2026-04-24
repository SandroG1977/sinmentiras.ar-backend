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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.rag_store import rag_store
from app.services.question_cache_store import question_cache_store


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
    _INFOLEG_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

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

    def _infoleg_get(self, url: str, *, timeout: int = 30) -> requests.Response:
        """Hace requests a InfoLEG con headers de navegador y un retry ante 403."""

        response = requests.get(
            url,
            timeout=timeout,
            headers=self._INFOLEG_HEADERS,
            allow_redirects=True,
        )
        if response.status_code == 403:
            # Retry simple para reducir falsos positivos por rate-limit/WAF.
            response = requests.get(
                url,
                timeout=timeout,
                headers=self._INFOLEG_HEADERS,
                allow_redirects=True,
            )
        response.raise_for_status()
        return response

    # Resolución y armado de URLs de InfoLEG.
    def _buscar_id_norma_infoleg(
        self,
        numero_ley: int,
        tipo_norma: int = 1,
        anio_sancion: str = "",
    ) -> tuple[int, str, str | None]:
        """Busca id interno de InfoLEG y resumen oficial desde buscarNormas HTML."""

        search_url = (
            f"{self._INFOLEG_BASE}/buscarNormas.do?tipoNorma={tipo_norma}"
            f"&numero={numero_ley}&anioSancion={anio_sancion}"
        )
        response = self._infoleg_get(search_url, timeout=20)
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

        infoleg_id = int(match.group(1))
        official_summary = self._extract_infoleg_summary_from_search_html(
            response.text,
            numero_ley=numero_ley,
            infoleg_id=infoleg_id,
        )
        return infoleg_id, search_url, official_summary

    @staticmethod
    def _extract_infoleg_summary_from_search_html(
        search_html: str,
        *,
        numero_ley: int,
        infoleg_id: int,
    ) -> str | None:
        """Extrae la descripción oficial de la fila de resultados en buscarNormas."""

        try:
            soup = BeautifulSoup(search_html, "html.parser")
            target_anchor = None
            for anchor in soup.find_all("a", href=True):
                href = str(anchor.get("href", ""))
                if "verNorma.do" not in href:
                    continue
                if f"id={infoleg_id}" in href:
                    target_anchor = anchor
                    break

            if target_anchor is None:
                # Fallback por texto de la ley si no se encontró por id.
                needle = f"{numero_ley}"
                for anchor in soup.find_all("a", href=True):
                    if "verNorma.do" not in str(anchor.get("href", "")):
                        continue
                    if needle in anchor.get_text(" ", strip=True):
                        target_anchor = anchor
                        break

            if target_anchor is None:
                return None

            row = target_anchor.find_parent("tr")
            if row is None:
                return None

            cells = row.find_all("td")
            if len(cells) < 2:
                return None

            # Estructura habitual: [identificador/norma, fecha, descripcion].
            # Elegimos la celda con más contenido textual entre las celdas no-identificador.
            candidates = [
                re.sub(r"\s+", " ", td.get_text(" ", strip=True)).strip()
                for td in cells[1:]
            ]
            candidates = [text for text in candidates if text]
            if not candidates:
                return None

            best = max(candidates, key=len)
            return best or None
        except Exception:
            return None

    def _url_ver_norma(self, infoleg_id: int) -> str:
        """Construye la URL de verNorma para una norma de InfoLEG."""

        return f"{self._INFOLEG_BASE}/verNorma.do?id={infoleg_id}"

    def _url_texact_fallback(self, infoleg_id: int) -> str:
        """Genera una URL de texto actualizado cuando no se pudo descubrir desde HTML."""

        low = (infoleg_id // 5000) * 5000
        high = low + 4999
        return f"{self._INFOLEG_ATTACH_BASE}/{low}-{high}/{infoleg_id}/texact.htm"

    def _url_norma_fallback(self, infoleg_id: int) -> str:
        """Genera una URL de norma base cuando no se pudo inferir desde texact."""

        low = (infoleg_id // 5000) * 5000
        high = low + 4999
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
        low = (infoleg_id // 5000) * 5000
        high = low + 4999
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
    @staticmethod
    def _build_law_hashtag(law_id: int, title_text: str) -> str:
        """Genera hashtag canonico para identificar rapidamente una ley en el catalogo."""

        base = re.sub(r"[^a-z0-9]+", "", title_text.lower())[:24]
        if not base:
            base = "ley"
        return f"#ley{law_id}{base}"

    @staticmethod
    def _extract_promulgated_on(texto_fuente: str) -> str | None:
        """Extrae fecha de promulgacion cuando aparece en el texto de la norma."""

        match = re.search(
            r"promulgad[ao]\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            texto_fuente,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _derive_law_title(
        law_id: int,
        soup_title: str | None,
        texto_fuente: str,
    ) -> str:
        """Obtiene un título útil de la ley, evitando títulos genéricos del portal."""

        raw_title = (soup_title or "").strip()
        low_title = raw_title.lower()

        # Si el título HTML es específico, lo conservamos.
        if raw_title and "infoleg" not in low_title and "ministerio" not in low_title:
            return raw_title

        # Intenta extraer el epígrafe legal desde el cuerpo de texto.
        text = re.sub(r"\s+", " ", texto_fuente).strip()
        pattern = re.compile(
            rf"ley\s+{law_id}\s*(.+?)(?:\.\s*sancionad[ao]|\.\s*promulgad[ao]|\s+sancionad[ao]:|\s+promulgad[ao]:)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .-:\t")
            if candidate:
                return f"Ley {law_id} - {candidate}"[:240]

        return f"Ley {law_id}"

    @staticmethod
    def _extract_keywords_from_chunks(
        law_id: int,
        title_text: str,
        chunks_estructurados: list[dict[str, object]],
        max_keywords: int = 12,
    ) -> list[str]:
        """Deriva keywords compactas para facilitar identificacion rapida de la ley."""

        stop = {
            "ley",
            "art",
            "articulo",
            "artículo",
            "titulo",
            "título",
            "capitulo",
            "capítulo",
            "seccion",
            "sección",
            "de",
            "del",
            "la",
            "el",
            "los",
            "las",
            "y",
            "en",
            "para",
            "por",
            "con",
            "que",
            "sobre",
            "norma",
        }

        bag = f"{title_text} " + " ".join(
            str(chunk.get("title", "")) for chunk in chunks_estructurados[:40]
        )
        tokens = re.findall(r"[a-záéíóúñ0-9]{4,}", bag.lower())

        counts: dict[str, int] = {}
        for token in tokens:
            if token in stop:
                continue
            counts[token] = counts.get(token, 0) + 1

        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        keywords = [f"ley-{law_id}"]
        keywords.extend([term for term, _count in ordered[: max_keywords - 1]])
        return keywords[:max_keywords]

    @staticmethod
    def _generate_law_summary(
        title: str,
        chunks_estructurados: list[dict[str, object]],
        max_chunks: int = 30,
    ) -> str | None:
        """Genera un resumen breve de la ley usando el LLM, a partir de sus primeros chunks.

        Se ejecuta una sola vez al ingestar. El resultado se persiste en rag_laws.summary_text.
        """

        if not settings.OPENAI_API_KEY:
            return None

        try:
            sample = chunks_estructurados[:max_chunks]
            context_lines = []
            for chunk in sample:
                chunk_title = str(chunk.get("title", "")).strip()
                chunk_text = str(chunk.get("text", "")).strip()
                if chunk_title:
                    context_lines.append(f"[{chunk_title}] {chunk_text}")
                else:
                    context_lines.append(chunk_text)
            context_block = "\n\n".join(context_lines)

            prompt = PromptTemplate.from_template(
                "Tenés el texto de la ley argentina '{titulo}'. "
                "Redactá un resumen claro de 3 a 5 oraciones para público general que explique:\n"
                "- De qué trata la ley\n"
                "- A quiénes aplica\n"
                "- Los 2 o 3 temas principales que regula\n\n"
                "Usá SOLO la información del contexto provisto. "
                "Escribí en español neutro, sin tecnicismos innecesarios.\n\n"
                "Contexto:\n{contexto}"
            )
            llm = ChatOpenAI(
                model=settings.OPENAI_QUERY_MODEL,
                api_key=lambda: settings.OPENAI_API_KEY or "",
                temperature=0,
            )
            result = (prompt | llm | StrOutputParser()).invoke(
                {"titulo": title, "contexto": context_block}
            )
            cleaned = result.strip()
            return cleaned if cleaned else None
        except Exception:
            return None

    def ingest_law_from_infoleg(
        self,
        law_id: int,
        *,
        replace_existing: bool = True,
    ) -> InfolegIngestResult:
        """Ingiere una ley desde InfoLEG, la fragmenta semánticamente y la persiste."""

        infoleg_id, search_url, official_summary = self._buscar_id_norma_infoleg(law_id)
        ver_norma_url = self._url_ver_norma(infoleg_id)

        ver_response = self._infoleg_get(ver_norma_url, timeout=20)
        ver_response.encoding = ver_response.apparent_encoding or "utf-8"

        texact_url = self._extraer_url_texact_desde_ver_norma(
            ver_response.text,
            ver_norma_url,
        ) or self._url_texact_fallback(infoleg_id)

        texact_response = self._infoleg_get(texact_url, timeout=30)
        texact_response.encoding = texact_response.apparent_encoding or "utf-8"

        sha256_hash = hashlib.sha256(texact_response.content).hexdigest()
        soup = BeautifulSoup(texact_response.text, "html.parser")
        texto_fuente_directo = soup.get_text(" ", strip=True)
        texto_fuente = re.sub(r"\s+", " ", texto_fuente_directo).strip()

        # Si texact.htm devuelve error, intentar con norma.htm como fallback.
        if not texto_fuente or self._is_infoleg_error_page(texto_fuente):
            norma_url = self._url_norma_fallback(infoleg_id)
            if norma_url != texact_url:
                norma_response = self._infoleg_get(norma_url, timeout=30)
                norma_response.encoding = norma_response.apparent_encoding or "utf-8"
                sha256_hash = hashlib.sha256(norma_response.content).hexdigest()
                soup = BeautifulSoup(norma_response.text, "html.parser")
                texto_fuente_directo = soup.get_text(" ", strip=True)
                texto_fuente = re.sub(r"\s+", " ", texto_fuente_directo).strip()
                texact_url = norma_url  # usar norma.htm como source_uri

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

        title_text = self._derive_law_title(
            law_id,
            soup.title.get_text(strip=True) if soup.title else None,
            texto_fuente,
        )

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

        law_summary = official_summary or self._generate_law_summary(
            title_text,
            chunks_estructurados,
        )

        rag_store.upsert_law_catalog_entry(
            law_id=law_id,
            law_number=str(law_id),
            title=title_text,
            hash_tag=self._build_law_hashtag(law_id, title_text),
            source_link=source_uri,
            promulgated_on=self._extract_promulgated_on(texto_fuente),
            keywords=self._extract_keywords_from_chunks(
                law_id,
                title_text,
                chunks_estructurados,
            ),
            summary_text=law_summary,
            last_document_id=document_id,
        )

        question_cache_store.invalidate_by_law_id(law_id)
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

    def _semantic_rank(
        self,
        query: str,
        pool_k: int,
    ) -> list[tuple[int, float]]:
        """Obtiene ranking semántico como pares (index_chunk, score)."""

        if self._faiss_index is None or not self._vectors or pool_k <= 0:
            return []

        try:
            import numpy as np

            query_vector = embedding_service.embed_texts([query])[0]
            matrix = np.array([query_vector], dtype="float32")
            scores, indexes = self._faiss_index.search(matrix, pool_k)
        except Exception:
            return []

        ranked: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indexes[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            ranked.append((int(idx), float(score)))
        return ranked

    def _lexical_rank(
        self,
        query_tokens: set[str],
        pool_k: int,
    ) -> list[tuple[int, float]]:
        """Obtiene ranking léxico como pares (index_chunk, score)."""

        if not query_tokens or pool_k <= 0:
            return []

        scored_chunks: list[tuple[float, int]] = []
        for idx, chunk in enumerate(self._chunks):
            chunk_tokens = self._tokenize(chunk.text)
            if not chunk_tokens:
                continue

            overlap = len(query_tokens.intersection(chunk_tokens))
            if overlap == 0:
                continue

            score = overlap / math.sqrt(len(chunk_tokens))
            scored_chunks.append((score, idx))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        selected = scored_chunks[:pool_k]
        return [(idx, score) for score, idx in selected]

    @staticmethod
    def _fuse_rrf(
        semantic_rank: list[tuple[int, float]],
        lexical_rank: list[tuple[int, float]],
        top_k: int,
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """Fusiona rankings con Reciprocal Rank Fusion (RRF)."""

        if top_k <= 0:
            return []

        rrf_scores: dict[int, float] = {}
        semantic_positions = {
            idx: pos for pos, (idx, _score) in enumerate(semantic_rank, start=1)
        }
        lexical_positions = {
            idx: pos for pos, (idx, _score) in enumerate(lexical_rank, start=1)
        }

        for idx in semantic_positions:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (
                k + semantic_positions[idx]
            )
        for idx in lexical_positions:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (
                k + lexical_positions[idx]
            )

        semantic_score_map = {idx: score for idx, score in semantic_rank}
        lexical_score_map = {idx: score for idx, score in lexical_rank}

        fused = sorted(
            rrf_scores.items(),
            key=lambda item: (
                item[1],
                semantic_score_map.get(item[0], float("-inf")),
                lexical_score_map.get(item[0], float("-inf")),
            ),
            reverse=True,
        )
        return fused[:top_k]

    def retrieve(self, query: str, top_k: int) -> list[dict[str, object]]:
        """Recupera chunks con búsqueda híbrida (semántica + léxica) y fusión RRF."""

        if not settings.RAG_ENABLED:
            return []

        if top_k <= 0:
            return []

        self._load_chunks()
        if not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        pool_k = min(len(self._chunks), max(top_k * 3, top_k))
        semantic_rank = self._semantic_rank(query, pool_k)
        lexical_rank = self._lexical_rank(query_tokens, pool_k)

        if semantic_rank and lexical_rank:
            fused = self._fuse_rrf(semantic_rank, lexical_rank, top_k)
            return [
                {
                    "text": self._chunks[idx].text,
                    "source": self._chunks[idx].source,
                    "score": round(score, 6),
                    "metadata": self._chunks[idx].metadata,
                }
                for idx, score in fused
            ]

        if semantic_rank:
            selected = semantic_rank[:top_k]
            return [
                {
                    "text": self._chunks[idx].text,
                    "source": self._chunks[idx].source,
                    "score": round(score, 4),
                    "metadata": self._chunks[idx].metadata,
                }
                for idx, score in selected
            ]

        if lexical_rank:
            selected = lexical_rank[:top_k]
            return [
                {
                    "text": self._chunks[idx].text,
                    "source": self._chunks[idx].source,
                    "score": round(score, 4),
                    "metadata": self._chunks[idx].metadata,
                }
                for idx, score in selected
            ]

        return []


rag_service = LocalRAGService()
