import json
import sqlite3
from pathlib import Path
from threading import Lock

from app.core.config import settings


class RAGStore:
    def __init__(self) -> None:
        self._lock = Lock()

    @property
    def db_path(self) -> str:
        return settings.RAG_DB_PATH

    def _connect(self) -> sqlite3.Connection:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(db_file))

    @property
    def _use_postgres(self) -> bool:
        return bool(settings.RAG_POSTGRES_DSN)

    def _connect_postgres(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("psycopg is required for PostgreSQL mode") from exc

        return psycopg.connect(settings.RAG_POSTGRES_DSN)

    def initialize(self) -> None:
        with self._lock:
            if self._use_postgres:
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
                    conn.commit()
                return

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

        if self._use_postgres:
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

        if self._use_postgres:
            with self._connect_postgres() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT chunk_text, source_uri, metadata_json FROM rag_chunks ORDER BY source_uri, chunk_index"
                    )
                    rows = cur.fetchall()
            return [
                {"text": row[0], "source": row[1], "metadata": row[2] or {}}
                for row in rows
            ]

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


rag_store = RAGStore()
