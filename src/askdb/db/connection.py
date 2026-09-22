import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class DatabaseError(Exception):
    """Raised when a database cannot be opened."""


def _decode(raw: bytes) -> str:
    # Benchmark databases contain values in mixed encodings; replacing bad bytes
    # keeps introspection working instead of failing the whole query.
    return raw.decode("utf-8", errors="replace")


@contextmanager
def read_only(path: Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite database that cannot be written to.

    Read-only is enforced by the driver rather than by convention, so a
    generated query containing INSERT or DROP fails at execution time.
    """
    if not path.is_file():
        raise DatabaseError(f"database not found: {path}")

    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.text_factory = _decode
    try:
        yield conn
    finally:
        conn.close()


def quote(identifier: str) -> str:
    """Quote a table or column name for safe interpolation."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'
