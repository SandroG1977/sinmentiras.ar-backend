import json
import sqlite3
from pathlib import Path
from threading import Lock
from time import time

from app.core.config import settings


class RAGStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._postgres_available: bool | None = None  # None = not checked yet
        self._postgres_check_time = 0.0
        self._postgres_check_interval = 30  # Retry every 30 seconds

    @property
    def db_path(self) -> str:
        return settings.RAG_DB_PATH

    def _connect(self) -> sqlite3.Connection:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(db_file))

    @property
    def _use_postgres_configured(self) -> bool:
        return bool(settings.RAG_POSTGRES_DSN)

    def _is_postgres_available(self) -> bool:
        """Check if Postgres is reachable. Caches result for _postgres_check_interval seconds."""
        now = time()

        # Use cached result if still valid
        if (
            self._postgres_available is not None
            and (now - self._postgres_check_time) < self._postgres_check_interval
        ):
            return self._postgres_available

        self._postgres_check_time = now

        if not self._use_postgres_configured:
            self._postgres_available = False
            return False

        try:
            import psycopg

            with psycopg.connect(settings.RAG_POSTGRES_DSN, timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            self._postgres_available = True
            return True
        except Exception:
            self._postgres_available = False
            return False

    def _connect_postgres(self):
        """Connect to Postgres, raise if not available."""
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("psycopg is required for PostgreSQL mode") from exc

        return psycopg.connect(settings.RAG_POSTGRES_DSN)

    def initialize(self) -> None:
        with self._lock:
            # Try Postgres first if configured
            if self._use_postgres_configured and self._is_postgres_available():
                try:
                    with self._connect_postgres() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                CREATE TABLE IF NOT EXISTS rag_chunks (
                                    id BIGSERIAL PRIMARY KEY,
                                    document_id TEXT NOT NULL,
                                    source_uri TEXT NOT NULL,
                                    chunk_index INTEGER NOT NULL,
                                    chunk_text TEXT NOT NULL,
                                    metadata_json JSONB,
                                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                                )
                                """
                            )
                            cur.execute(
                                "CREATE INDEX IF NOT EXISTS idx_rag_chunks_source_uri ON rag_chunks(source_uri)"
                            )
                            cur.execute(
                                "CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id ON rag_chunks(document_id)"
                            )
                            cur.execute(
                                """
                                CREATE TABLE IF NOT EXISTS rag_embeddings (
                                    id BIGSERIAL PRIMARY KEY,
                                    text_hash TEXT NOT NULL UNIQUE,
                                    embedding TEXT NOT NULL,
                                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                                )
                                """
                            )
                            cur.execute(
                                "CREATE INDEX IF NOT EXISTS idx_rag_embeddings_text_hash ON rag_embeddings(text_hash)"
                            )
                        conn.commit()
                    return
                except Exception:
                    # Fallback to SQLite if Postgres fails
                    pass

            # Fallback to SQLite
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL,
                        source_uri TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        chunk_text TEXT NOT NULL,
                        metadata_json TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_rag_chunks_source_uri ON rag_chunks(source_uri)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id ON rag_chunks(document_id)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text_hash TEXT NOT NULL UNIQUE,
                        embedding TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_rag_embeddings_text_hash ON rag_embeddings(text_hash)"
                )

    def ingest_document(
        self,
        *,
        document_id: str,
        source_uri: str,
        chunks: list[str],
        metadata: dict[str, str | int | float | bool | list[str] | None] | None,
        replace_existing: bool,
    ) -> int:
        chunk_records = [{"text": chunk, "metadata": {}} for chunk in chunks]
        return self.ingest_chunk_records(
            document_id=document_id,
            source_uri=source_uri,
            chunk_records=chunk_records,
            base_metadata=metadata,
            replace_existing=replace_existing,
        )

    def ingest_chunk_records(
        self,
        *,
        document_id: str,
        source_uri: str,
        chunk_records: list[dict[str, object]],
        base_metadata: dict[str, str | int | float | bool | list[str] | None] | None,
        replace_existing: bool,
    ) -> int:
        self.initialize()

        rows_to_insert: list[tuple[int, str, str]] = []
        for index, chunk_record in enumerate(chunk_records):
            text = str(chunk_record.get("text", "")).strip()
            if not text:
                continue

            metadata = dict(base_metadata or {})
            record_metadata = chunk_record.get("metadata")
            if isinstance(record_metadata, dict):
                metadata.update(record_metadata)

            rows_to_insert.append(
                (index, text, json.dumps(metadata, ensure_ascii=True))
            )

        if not rows_to_insert:
            return 0

        # Try Postgres first if configured and available
        if self._use_postgres_configured and self._is_postgres_available():
            try:
                with self._lock:
                    with self._connect_postgres() as conn:
                        with conn.cursor() as cur:
                            if replace_existing:
                                cur.execute(
                                    "DELETE FROM rag_chunks WHERE source_uri = %s",
                                    (source_uri,),
                                )

                            cur.executemany(
                                """
                                INSERT INTO rag_chunks(document_id, source_uri, chunk_index, chunk_text, metadata_json)
                                VALUES(%s, %s, %s, %s, %s::jsonb)
                                """,
                                [
                                    (
                                        document_id,
                                        source_uri,
                                        index,
                                        chunk_text,
                                        metadata_json,
                                    )
                                    for index, chunk_text, metadata_json in rows_to_insert
                                ],
                            )
                        conn.commit()
                return len(rows_to_insert)
            except Exception:
                # Fallback to SQLite
                pass

        # Fallback to SQLite
        with self._lock:
            with self._connect() as conn:
                if replace_existing:
                    conn.execute(
                        "DELETE FROM rag_chunks WHERE source_uri = ?", (source_uri,)
                    )

                conn.executemany(
                    """
                    INSERT INTO rag_chunks(document_id, source_uri, chunk_index, chunk_text, metadata_json)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    [
                        (document_id, source_uri, index, chunk_text, metadata_json)
                        for index, chunk_text, metadata_json in rows_to_insert
                    ],
                )
                conn.commit()

        return len(rows_to_insert)

    def list_chunks(self) -> list[dict[str, object]]:
        self.initialize()

        # Try Postgres first
        if self._use_postgres_configured and self._is_postgres_available():
            try:
                with self._connect_postgres() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT chunk_text, source_uri, metadata_json FROM rag_chunks ORDER BY source_uri, chunk_index"
                        )
                        rows = cur.fetchall()
                return [
                    {
                        "text": row[0],
                        "source": row[1],
                        "metadata": json.loads(row[2]) if row[2] else {},
                    }
                    for row in rows
                ]
            except Exception:
                pass

        # Fallback to SQLite
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT chunk_text, source_uri, metadata_json FROM rag_chunks ORDER BY source_uri, chunk_index"
                ).fetchall()

            chunks: list[dict[str, object]] = []
            for row in rows:
                raw_metadata = row[2] or "{}"
                try:
                    parsed_metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    parsed_metadata = {}
                chunks.append(
                    {"text": row[0], "source": row[1], "metadata": parsed_metadata}
                )
            return chunks
        except Exception:
            return []

    def cache_embedding(self, text_hash: str, embedding: list[float]) -> None:
        """Cache an embedding vector by text hash. Tries Postgres, falls back to SQLite."""
        self.initialize()

        embedding_json = json.dumps(embedding)

        # Try Postgres first
        if self._use_postgres_configured and self._is_postgres_available():
            with self._lock:
                try:
                    with self._connect_postgres() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO rag_embeddings(text_hash, embedding)
                                VALUES(%s, %s)
                                ON CONFLICT(text_hash) DO UPDATE SET embedding = EXCLUDED.embedding
                                """,
                                (text_hash, embedding_json),
                            )
                        conn.commit()
                    return
                except Exception:
                    pass

        # Fallback to SQLite
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO rag_embeddings(text_hash, embedding)
                        VALUES(?, ?)
                        """,
                        (text_hash, embedding_json),
                    )
                    conn.commit()
            except Exception:
                pass

    def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        """Retrieve cached embedding by text hash. Tries Postgres, falls back to SQLite."""
        self.initialize()

        # Try Postgres first
        if self._use_postgres_configured and self._is_postgres_available():
            try:
                with self._connect_postgres() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT embedding FROM rag_embeddings WHERE text_hash = %s",
                            (text_hash,),
                        )
                        row = cur.fetchone()
                return json.loads(row[0]) if row else None
            except Exception:
                pass

        # Fallback to SQLite
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT embedding FROM rag_embeddings WHERE text_hash = ?",
                    (text_hash,),
                ).fetchone()
            return json.loads(row[0]) if row else None
        except Exception:
            return None

    def get_latest_law_status(self, law_id: int) -> dict[str, object]:
        """Return whether a law is ingested and metadata for the most recently ingested version."""

        self.initialize()

        # Try Postgres first.
        if self._use_postgres_configured and self._is_postgres_available():
            try:
                with self._connect_postgres() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT document_id, source_uri, metadata_json, created_at
                            FROM rag_chunks
                            WHERE (metadata_json->>'law_id')::int = %s
                            ORDER BY created_at DESC, id DESC
                            LIMIT 1
                            """,
                            (law_id,),
                        )
                        latest_row = cur.fetchone()

                        cur.execute(
                            """
                            SELECT COUNT(DISTINCT document_id)
                            FROM rag_chunks
                            WHERE (metadata_json->>'law_id')::int = %s
                            """,
                            (law_id,),
                        )
                        versions = cur.fetchone()

                if latest_row:
                    metadata = latest_row[2] if isinstance(latest_row[2], dict) else {}
                    return {
                        "law_id": law_id,
                        "exists": True,
                        "versions_loaded": int(versions[0] if versions else 1),
                        "last_ingested_at": (
                            latest_row[3].isoformat() if latest_row[3] else None
                        ),
                        "latest_document_id": latest_row[0],
                        "latest_source_uri": latest_row[1],
                        "latest_source_actualizado": metadata.get("source_actualizado"),
                        "latest_source_norma": metadata.get("source_norma"),
                        "latest_sha256_hash": metadata.get("sha256_hash"),
                    }
            except Exception:
                pass

        # SQLite fallback.
        try:
            with self._connect() as conn:
                latest_row = conn.execute(
                    """
                    SELECT document_id, source_uri, metadata_json, created_at
                    FROM rag_chunks
                    WHERE CAST(json_extract(metadata_json, '$.law_id') AS INTEGER) = ?
                    ORDER BY datetime(created_at) DESC, id DESC
                    LIMIT 1
                    """,
                    (law_id,),
                ).fetchone()

                versions_row = conn.execute(
                    """
                    SELECT COUNT(DISTINCT document_id)
                    FROM rag_chunks
                    WHERE CAST(json_extract(metadata_json, '$.law_id') AS INTEGER) = ?
                    """,
                    (law_id,),
                ).fetchone()

            if latest_row:
                raw_metadata = latest_row[2] or "{}"
                try:
                    metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    metadata = {}

                return {
                    "law_id": law_id,
                    "exists": True,
                    "versions_loaded": int(versions_row[0] if versions_row else 1),
                    "last_ingested_at": latest_row[3],
                    "latest_document_id": latest_row[0],
                    "latest_source_uri": latest_row[1],
                    "latest_source_actualizado": metadata.get("source_actualizado"),
                    "latest_source_norma": metadata.get("source_norma"),
                    "latest_sha256_hash": metadata.get("sha256_hash"),
                }
        except Exception:
            pass

        return {
            "law_id": law_id,
            "exists": False,
            "versions_loaded": 0,
            "last_ingested_at": None,
            "latest_document_id": None,
            "latest_source_uri": None,
            "latest_source_actualizado": None,
            "latest_source_norma": None,
            "latest_sha256_hash": None,
        }


rag_store = RAGStore()
