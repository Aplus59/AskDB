"""On-disk cache of model responses.

Two reasons this exists, and the second is the important one.

The free Gemini tier allows 10-15 requests per minute, so re-running an
evaluation over 500 questions costs hours. Cached responses make repeated runs
effectively free, which is what makes iteration possible at all.

More importantly, an evaluation that calls the model afresh every time is not
reproducible: two runs of identical code produce different numbers, and there
is no way to tell a real improvement from sampling noise. Caching pins the
model's output so that a change in the score is a change you made.
"""

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from askdb.llm.types import Completion, Usage

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key           TEXT PRIMARY KEY,
    model         TEXT NOT NULL,
    text          TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at    REAL NOT NULL
)
"""


def cache_key(model: str, prompt: str, params: dict[str, Any] | None = None) -> str:
    """A stable identity for one model call.

    Parameters are sorted so that two calls differing only in dictionary order
    share a cache entry.
    """
    payload = json.dumps(
        {"model": model, "prompt": prompt, "params": params or {}},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str) -> Completion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT model, text, input_tokens, output_tokens FROM responses WHERE key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        return Completion(
            text=str(row["text"]),
            model=str(row["model"]),
            usage=Usage(
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
            ),
            cached=True,
        )

    def put(self, key: str, completion: Completion) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(key, model, text, input_tokens, output_tokens, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    completion.model,
                    completion.text,
                    completion.usage.input_tokens,
                    completion.usage.output_tokens,
                    time.time(),
                ),
            )

    def __len__(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM responses").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM responses")
