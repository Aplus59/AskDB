"""Turning a schema into searchable text.

Retrieval decides which tables reach the prompt, so it decides what the model
is able to answer at all. Getting the text representation right matters more
than the ranking function applied on top of it.

Real schema identifiers are hostile to matching: `sch_name`, `FreeMealCount`,
`Free Meal Count (K-12)`. All three have to reduce to tokens a question can
match against.
"""

import re
from dataclasses import dataclass

from askdb.db.catalog import Catalog, Table

# Split camelCase, and also split an acronym from a word that follows it so
# that BIRD's `CDSCode` becomes "cds code" rather than one opaque token.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALPHANUMERIC = re.compile(r"[^A-Za-z0-9]+")

DEFAULT_TABLE_NAME_WEIGHT = 2


def tokenize(text: str) -> list[str]:
    """Reduce an identifier or a question to lowercase tokens."""
    spaced = _CAMEL_BOUNDARY.sub(" ", text)
    return [part.lower() for part in _NON_ALPHANUMERIC.split(spaced) if part]


@dataclass(frozen=True)
class TableDocument:
    table: str
    tokens: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.tokens)


def describe(table: Table, table_name_weight: int = DEFAULT_TABLE_NAME_WEIGHT) -> TableDocument:
    """Build the searchable token list for one table.

    The table name is repeated rather than scored in a separate field. BM25
    already rewards term frequency, so repetition is the cheapest way to weight
    it, and it keeps the ranking function unmodified.

    Foreign key targets are included because questions that need a join often
    name the other table rather than this one.
    """
    name_tokens = tokenize(table.name)
    tokens: list[str] = []
    tokens.extend(name_tokens * table_name_weight)

    for column in table.columns:
        tokens.extend(tokenize(column.name))

    for foreign_key in table.foreign_keys:
        tokens.extend(tokenize(foreign_key.references_table))

    return TableDocument(table=table.name, tokens=tuple(tokens))


def describe_catalog(
    catalog: Catalog, table_name_weight: int = DEFAULT_TABLE_NAME_WEIGHT
) -> tuple[TableDocument, ...]:
    return tuple(describe(table, table_name_weight) for table in catalog.tables)
