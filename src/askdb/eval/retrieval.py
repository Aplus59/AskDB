"""Measuring schema retrieval.

Retrieval is scored by asking whether every table the reference query needed
appears in the top k results. If it does not, the agent is being asked to
write a query against a schema it was never shown, and no amount of prompting
recovers from that.

Gold tables are recovered without parsing SQL. A parser would be a dependency
and a source of dialect bugs, and two cheaper rules get the same answer here:
only read the FROM and JOIN clauses, then keep only names the catalog knows.
The first rule stops a column named `artist` from counting as the `artist`
table; the second stops a CTE or alias from counting as anything.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")

# The body of a FROM or JOIN clause, ending at whatever clause comes next.
_TABLE_CLAUSE = re.compile(
    r"\b(?:from|join)\b"
    r"(?P<body>.*?)"
    r"(?=\b(?:where|group|order|having|limit|on|using|union|intersect|except|select|from|join)\b"
    r"|\)|$)",
    re.IGNORECASE | re.DOTALL,
)

_IDENTIFIER = re.compile(r"`([^`]+)`|\"([^\"]+)\"|\[([^\]]+)\]|\b([A-Za-z_]\w*)\b")


def _strip_noise(sql: str) -> str:
    without_comments = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))
    return _STRING_LITERAL.sub(" ", without_comments)


def identifiers_in(text: str) -> list[str]:
    """Every identifier in a fragment, with any quoting removed."""
    found = []
    for match in _IDENTIFIER.finditer(text):
        backtick, double_quoted, bracketed, bare = match.groups()
        found.append(backtick or double_quoted or bracketed or bare)
    return found


def table_clauses(sql: str) -> list[str]:
    """The text following each FROM or JOIN, where table names live."""
    return [match.group("body") for match in _TABLE_CLAUSE.finditer(_strip_noise(sql))]


def tables_mentioned(sql: str, known_tables: Iterable[str]) -> frozenset[str]:
    """Which of the database's tables this query reads from.

    Returns the catalog's own spelling of each name, so the result is directly
    comparable with what the retriever ranks, whatever case the query used.
    """
    by_lowercase = {name.lower(): name for name in known_tables}
    mentioned = {
        by_lowercase[identifier.lower()]
        for clause in table_clauses(sql)
        for identifier in identifiers_in(clause)
        if identifier.lower() in by_lowercase
    }
    return frozenset(mentioned)


@dataclass(frozen=True)
class Outcome:
    """One question's retrieval result."""

    question_id: int
    gold_tables: frozenset[str]
    ranked: tuple[str, ...]

    def hit_at(self, k: int) -> bool:
        """Whether every gold table appears within the top k.

        All of them, not any: a query joining two tables fails if either one
        is missing from the prompt.
        """
        if not self.gold_tables:
            return False
        return self.gold_tables <= set(self.ranked[:k])

    def depth_needed(self) -> int | None:
        """The smallest k that would have retrieved every gold table.

        Useful for choosing k: the distribution of this value across a dataset
        says where to cut without reading recall curves.
        """
        if not self.gold_tables:
            return None
        remaining = set(self.gold_tables)
        for position, table in enumerate(self.ranked, start=1):
            remaining.discard(table)
            if not remaining:
                return position
        return None


@dataclass(frozen=True)
class Report:
    recall: dict[int, float]
    scored: int
    skipped: int

    def summary(self) -> str:
        parts = [f"recall@{k}={value:.3f}" for k, value in sorted(self.recall.items())]
        return f"{', '.join(parts)}  (n={self.scored}, skipped={self.skipped})"


def evaluate(outcomes: Sequence[Outcome], k_values: Sequence[int] = (1, 3, 5, 10)) -> Report:
    """Aggregate recall at several depths.

    Questions whose gold tables could not be determined are counted separately
    rather than scored as failures, so a gap in extraction never quietly
    depresses the headline number.
    """
    scored = [outcome for outcome in outcomes if outcome.gold_tables]
    skipped = len(outcomes) - len(scored)

    if not scored:
        return Report(recall=dict.fromkeys(k_values, 0.0), scored=0, skipped=skipped)

    recall = {k: sum(outcome.hit_at(k) for outcome in scored) / len(scored) for k in k_values}
    return Report(recall=recall, scored=len(scored), skipped=skipped)
