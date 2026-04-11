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

    def initialize(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS question_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question_text TEXT NOT NULL,
                        question_vector_json TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        used_model TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        hit_count INTEGER NOT NULL DEFAULT 0,
                        last_hit_at TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_question_cache_created ON question_cache(created_at)"
                )

    def find_best_answer(self, question: str, min_similarity: float) -> tuple[str, str, float] | None:
        if not settings.CACHE_ENABLED:
            return None

        self.initialize()
        query_vector = embedding_service.embed_texts([question])[0]

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

    def save_answer(self, question: str, answer: str, used_model: str) -> None:
        if not settings.CACHE_ENABLED:
            return

        self.initialize()
        vector = embedding_service.embed_texts([question])[0]

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO question_cache(question_text, question_vector_json, answer_text, used_model)
                VALUES(?, ?, ?, ?)
                """,
                (question, json.dumps(vector, ensure_ascii=True), answer, used_model),
            )
            conn.commit()


question_cache_store = QuestionCacheStore()
