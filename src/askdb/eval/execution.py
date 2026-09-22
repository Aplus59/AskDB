"""Execution accuracy.

A predicted query is judged by running it and comparing the rows it returns
against the rows the gold query returns. Comparing SQL text would be wrong:
there are many correct ways to write the same query.

Three decisions are baked in here, and each one changes the score:

1. Row order is ignored. BIRD's official metric compares result sets, so
   matching it keeps our numbers comparable to the published leaderboard.
2. The headline metric collapses duplicate rows, again to match the official
   metric. That can hide a real difference between COUNT and COUNT DISTINCT,
   so `exact_match` additionally compares rows as an ordered multiset and is
   reported alongside as a diagnostic.
3. Floats are rounded before comparison. Exact float equality makes the metric
   noisy for any query involving division or averages. This is a deliberate
   deviation from the official implementation and is recorded as such.
"""

import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from askdb.db.connection import read_only

FLOAT_PRECISION = 6
DEFAULT_TIMEOUT_SECONDS = 30.0

Row = tuple[Any, ...]


@dataclass(frozen=True)
class ExecutionResult:
    rows: tuple[Row, ...] | None
    error: str | None
    duration_ms: float
    timed_out: bool = False
    columns: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Comparison:
    """The outcome of scoring one predicted query against its gold query."""

    match: bool
    exact_match: bool
    predicted: ExecutionResult
    gold: ExecutionResult

    @property
    def failure_reason(self) -> str | None:
        if self.match:
            return None
        if self.predicted.timed_out:
            return "timeout"
        if not self.predicted.ok:
            return "execution_error"
        if not self.gold.ok:
            return "gold_failed"
        return "wrong_rows"


def _install_timeout(conn: sqlite3.Connection, seconds: float) -> None:
    deadline = time.monotonic() + seconds

    def interrupt_when_expired() -> int:
        return 1 if time.monotonic() > deadline else 0

    # The handler runs every N virtual machine instructions; returning non-zero
    # aborts the query. Without this a cross join can hang the whole eval run.
    conn.set_progress_handler(interrupt_when_expired, 1000)


def run(
    database: Path,
    sql: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_rows: int | None = None,
) -> ExecutionResult:
    """Execute a query and capture rows, or the error it produced.

    `max_rows` bounds what is returned, for callers showing a result to a
    model rather than scoring it. Scoring must never pass it: a truncated
    result compares unequal to a complete one, so `compare_rows` refuses
    truncated input rather than reporting a wrong answer.
    """
    started = time.perf_counter()
    try:
        with read_only(database) as conn:
            _install_timeout(conn, timeout_seconds)
            cursor = conn.execute(sql)
            columns = tuple(column[0] for column in cursor.description or ())

            if max_rows is None:
                fetched = cursor.fetchall()
                truncated = False
            else:
                # One extra row reveals whether more were waiting.
                fetched = cursor.fetchmany(max_rows + 1)
                truncated = len(fetched) > max_rows
                fetched = fetched[:max_rows]

            rows = tuple(tuple(row) for row in fetched)
    except sqlite3.Error as exc:
        elapsed = (time.perf_counter() - started) * 1000
        message = str(exc)
        # SQLite reports an aborted progress handler as a generic interrupt.
        timed_out = "interrupted" in message.lower()
        return ExecutionResult(
            rows=None,
            error="query timed out" if timed_out else message,
            duration_ms=elapsed,
            timed_out=timed_out,
        )

    return ExecutionResult(
        rows=rows,
        error=None,
        duration_ms=(time.perf_counter() - started) * 1000,
        columns=columns,
        truncated=truncated,
    )


def _normalize(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, FLOAT_PRECISION)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _normalize_rows(rows: tuple[Row, ...]) -> list[Row]:
    return [tuple(_normalize(value) for value in row) for row in rows]


def compare_rows(predicted: tuple[Row, ...], gold: tuple[Row, ...]) -> tuple[bool, bool]:
    """Return (set match, multiset match) for two result sets."""
    left = _normalize_rows(predicted)
    right = _normalize_rows(gold)
    return set(left) == set(right), Counter(left) == Counter(right)


def score(
    database: Path,
    predicted_sql: str,
    gold_sql: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Comparison:
    """Run both queries against the same database and compare their rows."""
    gold = run(database, gold_sql, timeout_seconds)
    predicted = run(database, predicted_sql, timeout_seconds)

    if predicted.truncated or gold.truncated:
        raise ValueError("cannot score truncated results")

    if predicted.rows is None or gold.rows is None:
        return Comparison(match=False, exact_match=False, predicted=predicted, gold=gold)

    match, exact = compare_rows(predicted.rows, gold.rows)
    return Comparison(match=match, exact_match=exact, predicted=predicted, gold=gold)
