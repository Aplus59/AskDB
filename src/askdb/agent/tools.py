"""The tools the agent can call.

Every tool returns a structured result that knows how to render itself. Tests
assert against the structure; the agent sends the rendering. Keeping those
apart means prompt wording can change without breaking the tests that prove
the tool is correct.

Two rules hold across all of them:

- Output is bounded. A database here reaches 600 MB, and a single unbounded
  SELECT would blow the context window and the token budget together.
- Nothing raises on bad input from the model. A wrong table name or invalid
  SQL comes back as a message the agent can read and act on, because
  recovering from its own mistakes is the loop's whole job.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from askdb.config import settings
from askdb.db.catalog import Catalog, Table, sample_values
from askdb.db.connection import read_only
from askdb.eval import execution
from askdb.retrieval.documents import tokenize

DEFAULT_MAX_ROWS = 50
DEFAULT_COLUMN_MATCHES = 12
MAX_CELL_WIDTH = 60

# Only text columns get example values. A filter on a number or a date is
# written from the question; a filter on a category has to match how the value
# is actually spelled, and that is not guessable from a column name.
TEXT_TYPE_MARKERS = ("CHAR", "TEXT", "CLOB", "STRING")
MAX_SAMPLE_WIDTH = 32


def is_text_column(column_type: str) -> bool:
    upper = column_type.upper()
    return not upper or any(marker in upper for marker in TEXT_TYPE_MARKERS)


def _truncate(value: Any) -> str:
    text = "NULL" if value is None else str(value)
    if len(text) > MAX_CELL_WIDTH:
        return text[: MAX_CELL_WIDTH - 1] + "…"
    return text


@dataclass(frozen=True)
class TableSummary:
    name: str
    row_count: int
    column_count: int


@dataclass(frozen=True)
class TableList:
    tables: tuple[TableSummary, ...]

    def render(self) -> str:
        if not self.tables:
            return "This database has no tables."
        lines = [
            f"{summary.name} ({summary.column_count} columns, {summary.row_count} rows)"
            for summary in self.tables
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class TableDescription:
    table: Table | None
    samples: dict[str, tuple[str, ...]]
    available: tuple[str, ...] = ()

    def render(self) -> str:
        if self.table is None:
            return (
                "No such table. Available tables: " + ", ".join(self.available)
                if self.available
                else "No such table."
            )

        lines = [f"TABLE {self.table.name} ({self.table.row_count} rows)"]
        for column in self.table.columns:
            flags = []
            if column.primary_key:
                flags.append("primary key")
            if not column.nullable:
                flags.append("not null")
            suffix = f"  [{', '.join(flags)}]" if flags else ""

            examples = self.samples.get(column.name, ())
            shown = f"  e.g. {', '.join(_truncate(v) for v in examples)}" if examples else ""
            lines.append(f"  {column.name} {column.type}{suffix}{shown}")

        for foreign_key in self.table.foreign_keys:
            lines.append(
                f"  FOREIGN KEY {foreign_key.column} -> "
                f"{foreign_key.references_table}.{foreign_key.references_column}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ColumnMatch:
    table: str
    column: str
    type: str
    score: float


@dataclass(frozen=True)
class ColumnMatches:
    term: str
    matches: tuple[ColumnMatch, ...]

    def render(self) -> str:
        if not self.matches:
            return f"No column resembles {self.term!r}."
        lines = [f"{m.table}.{m.column} ({m.type})" for m in self.matches]
        return "\n".join(lines)


@dataclass(frozen=True)
class QueryOutput:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    error: str | None = None
    truncated: bool = False
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def render(self) -> str:
        if self.error is not None:
            return f"Query failed: {self.error}"
        if not self.rows:
            return "Query returned no rows."

        header = " | ".join(self.columns)
        body = "\n".join(" | ".join(_truncate(cell) for cell in row) for row in self.rows)
        footer = f"\n({len(self.rows)} rows shown, more available)" if self.truncated else ""
        return f"{header}\n{body}{footer}"


def score_column(term_tokens: list[str], column_name: str) -> float:
    """How well a column name matches a search term.

    Deterministic and cheap on purpose: this runs across every column of every
    table on each call, and a model-based matcher here would cost more than
    the query it is helping to write.
    """
    column_tokens = tokenize(column_name)
    if not column_tokens or not term_tokens:
        return 0.0

    lowered = column_name.lower()
    joined = " ".join(term_tokens)
    if lowered == joined.replace(" ", "") or column_tokens == term_tokens:
        return 1.0

    overlap = len(set(term_tokens) & set(column_tokens))
    if overlap:
        return 0.5 + 0.4 * (overlap / len(set(term_tokens) | set(column_tokens)))

    if joined.replace(" ", "") in lowered.replace("_", ""):
        return 0.4
    return 0.0


class Toolbox:
    """The tools bound to one database."""

    def __init__(
        self, database: Path, catalog: Catalog, *, max_rows: int = DEFAULT_MAX_ROWS
    ) -> None:
        self.database = database
        self.catalog = catalog
        self.max_rows = max_rows
        self._context_cache: dict[tuple[tuple[str, ...], int], str] = {}

    def schema_context(
        self, tables: Sequence[str], *, samples_per_column: int = 3
    ) -> str:
        """The schema as shown to the model, with example values inline.

        A model that cannot see the data writes `WHERE region = 'East Bohemia'`
        against a column storing `east Bohemia`, and gets zero rows with no
        error to recover from. Showing a few real values costs no model call
        and removes that entire failure mode.

        Cached per table set: every question against a database would otherwise
        re-sample the same columns, and some of these tables are hundreds of
        megabytes.
        """
        key = (tuple(tables), samples_per_column)
        if key in self._context_cache:
            return self._context_cache[key]

        wanted = [name.lower() for name in tables]
        blocks = []
        with read_only(self.database) as conn:
            for table in self.catalog.tables:
                if table.name.lower() not in wanted:
                    continue
                blocks.append(self._render_table(conn, table, samples_per_column))

        rendered = "\n\n".join(blocks)
        self._context_cache[key] = rendered
        return rendered

    def _render_table(self, conn: Any, table: Table, samples_per_column: int) -> str:
        lines = []
        for column in table.columns:
            line = f"  {column.name} {column.type}"
            if column.primary_key:
                line += " PRIMARY KEY"

            if samples_per_column and is_text_column(column.type):
                try:
                    values = sample_values(
                        conn, table.name, column.name, limit=samples_per_column
                    )
                except Exception:  # noqa: BLE001 - examples are optional
                    values = ()
                usable = [v for v in values if v and len(v) <= MAX_SAMPLE_WIDTH]
                if usable:
                    line += "  -- e.g. " + ", ".join(repr(v) for v in usable)
            lines.append(line)

        for foreign_key in table.foreign_keys:
            lines.append(
                f"  FOREIGN KEY ({foreign_key.column}) REFERENCES "
                f"{foreign_key.references_table}({foreign_key.references_column})"
            )

        body = "\n".join(lines)
        return f"CREATE TABLE {table.name} (\n{body}\n);  -- {table.row_count} rows"

    def list_tables(self) -> TableList:
        return TableList(
            tables=tuple(
                TableSummary(
                    name=table.name,
                    row_count=table.row_count,
                    column_count=len(table.columns),
                )
                for table in self.catalog.tables
            )
        )

    def describe_table(self, name: str, *, samples_per_column: int = 3) -> TableDescription:
        table = self.catalog.table(name)
        if table is None:
            return TableDescription(
                table=None,
                samples={},
                available=tuple(t.name for t in self.catalog.tables),
            )

        samples: dict[str, tuple[str, ...]] = {}
        with read_only(self.database) as conn:
            for column in table.columns:
                try:
                    samples[column.name] = sample_values(
                        conn, table.name, column.name, limit=samples_per_column
                    )
                except Exception:  # noqa: BLE001 - samples are a convenience, not a requirement
                    samples[column.name] = ()

        return TableDescription(table=table, samples=samples)

    def search_columns(self, term: str, *, limit: int = DEFAULT_COLUMN_MATCHES) -> ColumnMatches:
        term_tokens = tokenize(term)
        found = [
            ColumnMatch(
                table=table.name,
                column=column.name,
                type=column.type,
                score=score_column(term_tokens, column.name),
            )
            for table in self.catalog.tables
            for column in table.columns
        ]
        ranked = [match for match in found if match.score > 0]
        ranked.sort(key=lambda m: (-m.score, m.table, m.column))
        return ColumnMatches(term=term, matches=tuple(ranked[:limit]))

    def sample_values(
        self, table: str, column: str, *, limit: int | None = None
    ) -> tuple[str, ...]:
        known = self.catalog.table(table)
        if known is None or known.column(column) is None:
            return ()
        with read_only(self.database) as conn:
            return sample_values(
                conn, known.name, column, limit=limit or settings.sample_values_limit
            )

    def execute_sql(self, sql: str) -> QueryOutput:
        result = execution.run(self.database, sql, max_rows=self.max_rows)
        return QueryOutput(
            columns=result.columns,
            rows=result.rows or (),
            error=result.error,
            truncated=result.truncated,
            duration_ms=result.duration_ms,
        )
