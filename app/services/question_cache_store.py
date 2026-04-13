import json
import sqlite3
from pathlib import Path
from threading import Lock

from app.core.config import settings
from app.services.embedding_service import embedding_service


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


class QuestionCacheStore:
    def __init__(self) -> None:
        self._lock = Lock()

    @property
    def db_path(self) -> str:
        return settings.CACHE_SQLITE_PATH

    def _connect(self) -> sqlite3.Connection:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(db_file))

    @property
    def _use_postgres(self) -> bool:
        return bool(settings.CACHE_POSTGRES_DSN)

    def _connect_postgres(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("psycopg is required for PostgreSQL mode") from exc

        return psycopg.connect(settings.CACHE_POSTGRES_DSN)

    def initialize(self) -> None:
        with self._lock:
            if self._use_postgres:
                with self._connect_postgres() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS question_cache (
                                id BIGSERIAL PRIMARY KEY,
                                question_text TEXT NOT NULL,
                                question_vector_json JSONB NOT NULL,
                                answer_text TEXT NOT NULL,
                                used_model TEXT NOT NULL,
                                law_ids_json JSONB,
                                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                hit_count INTEGER NOT NULL DEFAULT 0,
                                last_hit_at TIMESTAMPTZ
                            )
                            """
                        )
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_question_cache_created ON question_cache(created_at)"
                        )
                        cur.execute(
                            "ALTER TABLE question_cache ADD COLUMN IF NOT EXISTS law_ids_json JSONB"
                        )
                    conn.commit()
                return

            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS question_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question_text TEXT NOT NULL,
                        question_vector_json TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        used_model TEXT NOT NULL,
                        law_ids_json TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        hit_count INTEGER NOT NULL DEFAULT 0,
                        last_hit_at TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_question_cache_created ON question_cache(created_at)"
                )
                columns = {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(question_cache)"
                    ).fetchall()
                }
                if "law_ids_json" not in columns:
                    conn.execute(
                        "ALTER TABLE question_cache ADD COLUMN law_ids_json TEXT"
                    )

    def find_best_answer(
        self, question: str, min_similarity: float
    ) -> tuple[str, str, float] | None:
        if not settings.CACHE_ENABLED:
            return None

        self.initialize()
        query_vector = embedding_service.embed_texts([question])[0]

        if self._use_postgres:
            with self._connect_postgres() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, question_vector_json, answer_text, used_model FROM question_cache"
                    )
                    rows = cur.fetchall()

            best: tuple[int, str, str, float] | None = None
            for row_id, vector_json, answer_text, used_model in rows:
                try:
                    cached_vector = [float(value) for value in vector_json]
                except (TypeError, ValueError):
                    continue

                similarity = _cosine_similarity(query_vector, cached_vector)
                if similarity < min_similarity:
                    continue

                if best is None or similarity > best[3]:
                    best = (row_id, answer_text, used_model, similarity)

            if not best:
                return None

            with self._connect_postgres() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE question_cache
                        SET hit_count = hit_count + 1,
                            last_hit_at = NOW()
                        WHERE id = %s
                        """,
                        (best[0],),
                    )
                conn.commit()

            return best[1], best[2], best[3]

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, question_vector_json, answer_text, used_model FROM question_cache"
            ).fetchall()

        best: tuple[int, str, str, float] | None = None
        for row_id, vector_json, answer_text, used_model in rows:
            try:
                cached_vector = [float(value) for value in json.loads(vector_json)]
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

            similarity = _cosine_similarity(query_vector, cached_vector)
            if similarity < min_similarity:
                continue

            if best is None or similarity > best[3]:
                best = (row_id, answer_text, used_model, similarity)

        if not best:
            return None

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE question_cache
                SET hit_count = hit_count + 1,
                    last_hit_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (best[0],),
            )
            conn.commit()

        return best[1], best[2], best[3]

    def save_answer(
        self,
        question: str,
        answer: str,
        used_model: str,
        law_ids: list[int] | None = None,
    ) -> None:
        if not settings.CACHE_ENABLED:
            return

        self.initialize()
        vector = embedding_service.embed_texts([question])[0]
        law_ids_json = json.dumps(law_ids or [], ensure_ascii=True)

        if self._use_postgres:
            with self._connect_postgres() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO question_cache(question_text, question_vector_json, answer_text, used_model, law_ids_json)
                        VALUES(%s, %s::jsonb, %s, %s, %s::jsonb)
                        """,
                        (
                            question,
                            json.dumps(vector, ensure_ascii=True),
                            answer,
                            used_model,
                            law_ids_json,
                        ),
                    )
                conn.commit()
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO question_cache(question_text, question_vector_json, answer_text, used_model, law_ids_json)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    question,
                    json.dumps(vector, ensure_ascii=True),
                    answer,
                    used_model,
                    law_ids_json,
                ),
            )
            conn.commit()

    def invalidate_by_law_id(self, law_id: int) -> int:
        """Elimina entradas de cache asociadas a una ley especifica. Devuelve cantidad borrada."""

        self.initialize()

        if self._use_postgres:
            with self._connect_postgres() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM question_cache
                        WHERE law_ids_json @> %s::jsonb
                        """,
                        (json.dumps([law_id]),),
                    )
                    deleted = cur.rowcount
                conn.commit()
            return deleted

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, law_ids_json FROM question_cache WHERE law_ids_json IS NOT NULL"
            ).fetchall()

            ids_to_delete: list[int] = []
            for row_id, raw in rows:
                try:
                    parsed = json.loads(raw or "[]")
                except (json.JSONDecodeError, TypeError):
                    parsed = []
                if law_id in parsed:
                    ids_to_delete.append(row_id)

            if not ids_to_delete:
                return 0

            conn.executemany(
                "DELETE FROM question_cache WHERE id = ?",
                [(rid,) for rid in ids_to_delete],
            )
            conn.commit()

        return len(ids_to_delete)


question_cache_store = QuestionCacheStore()
