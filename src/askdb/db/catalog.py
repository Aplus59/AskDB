"""Schema introspection.

Everything downstream works from a Catalog: schema retrieval ranks tables,
the agent describes them, and the evaluator compares result rows. Loading it
correctly matters more than anything built on top of it.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from askdb.config import settings
from askdb.db.connection import quote, read_only


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool
    primary_key: bool


@dataclass(frozen=True)
class ForeignKey:
    column: str
    references_table: str
    references_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...]
    row_count: int

    def column(self, name: str) -> Column | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)


@dataclass(frozen=True)
class Catalog:
    name: str
    tables: tuple[Table, ...]

    def table(self, name: str) -> Table | None:
        lowered = name.lower()
        return next((t for t in self.tables if t.name.lower() == lowered), None)

    def to_ddl(self, tables: list[str] | None = None) -> str:
        """Render the schema as CREATE TABLE statements.

        Passing `tables` renders only those, which is how schema retrieval
        keeps large databases out of the prompt.
        """
        wanted = self.tables
        if tables is not None:
            lowered = {t.lower() for t in tables}
            wanted = tuple(t for t in self.tables if t.name.lower() in lowered)
        return "\n\n".join(_render_table(t) for t in wanted)


def _render_table(table: Table) -> str:
    parts = []
    for column in table.columns:
        line = f"  {column.name} {column.type}"
        if column.primary_key:
            line += " PRIMARY KEY"
        elif not column.nullable:
            line += " NOT NULL"
        parts.append(line)
    for fk in table.foreign_keys:
        parts.append(
            f"  FOREIGN KEY ({fk.column}) "
            f"REFERENCES {fk.references_table}({fk.references_column})"
        )
    body = ",\n".join(parts)
    return f"CREATE TABLE {table.name} (\n{body}\n);  -- {table.row_count} rows"


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _columns(conn: sqlite3.Connection, table: str) -> tuple[Column, ...]:
    rows = conn.execute(f"PRAGMA table_info({quote(table)})").fetchall()
    return tuple(
        Column(
            name=str(row["name"]),
            type=str(row["type"] or "").upper() or "UNKNOWN",
            nullable=not bool(row["notnull"]),
            primary_key=bool(row["pk"]),
        )
        for row in rows
    )


def _foreign_keys(
    conn: sqlite3.Connection, table: str, primary_keys: dict[str, str | None]
) -> tuple[ForeignKey, ...]:
    rows = conn.execute(f"PRAGMA foreign_key_list({quote(table)})").fetchall()
    keys = []
    for row in rows:
        target_table = str(row["table"])
        target_column = row["to"]
        if target_column is None:
            # SQLite leaves this empty when the reference points at the other
            # table's primary key without naming it.
            resolved = primary_keys.get(target_table.lower())
            if resolved is None:
                continue
            target_column = resolved
        keys.append(
            ForeignKey(
                column=str(row["from"]),
                references_table=target_table,
                references_column=str(target_column),
            )
        )
    return tuple(keys)


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {quote(table)}").fetchone()
    return int(row["n"])


def load(path: Path) -> Catalog:
    """Read the full schema of a SQLite database."""
    with read_only(path) as conn:
        names = _table_names(conn)
        columns = {name: _columns(conn, name) for name in names}
        primary_keys: dict[str, str | None] = {
            name.lower(): next((c.name for c in cols if c.primary_key), None)
            for name, cols in columns.items()
        }
        tables = tuple(
            Table(
                name=name,
                columns=columns[name],
                foreign_keys=_foreign_keys(conn, name, primary_keys),
                row_count=_row_count(conn, name),
            )
            for name in names
        )
    return Catalog(name=path.stem, tables=tables)


def sample_values(
    conn: sqlite3.Connection, table: str, column: str, limit: int | None = None
) -> tuple[str, ...]:
    """Distinct non-null values from a column.

    Used to match a filter mentioned in a question against how the value is
    actually spelled in the data.
    """
    rows = conn.execute(
        f"SELECT DISTINCT {quote(column)} AS value FROM {quote(table)} "
        f"WHERE {quote(column)} IS NOT NULL LIMIT ?",
        (limit or settings.sample_values_limit,),
    ).fetchall()
    return tuple(str(row["value"]) for row in rows)
