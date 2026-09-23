"""Noticing that a successful query probably answered the wrong question.

The repair loop only fires when the database rejects a query. Measurement
showed that almost never happens: the model writes valid SQL, and the failures
are queries that run perfectly and return the wrong thing. One measured case
returned zero rows for a "how many" question, because the filter was spelled
`'East Bohemia'` against data storing `'east Bohemia'` — valid SQL, no error,
nothing for the repair loop to see.

These checks are deliberately few and conservative. Every suspicion raised
costs a model call, so a heuristic that fires on healthy results spends real
money to make the system worse. Each one here describes a result that is
almost certainly not what a well-formed question was asking for.
"""

import re
from dataclasses import dataclass

from askdb.agent.tools import QueryOutput

EMPTY_RESULT = "empty_result"
NULL_SCALAR = "null_scalar"
EXPECTED_ONE_ROW = "expected_one_row"

# Questions whose answer is a single number. Anchored at the start so that
# "which schools have more than 500 students" is not mistaken for one.
SCALAR_QUESTION = re.compile(
    r"^\s*(how many|how much|what is the (total|number|count|average|sum|maximum|minimum)"
    r"|count |calculate the)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Suspicion:
    code: str
    message: str


def expects_single_value(question: str) -> bool:
    return SCALAR_QUESTION.search(question) is not None


def inspect(question: str, output: QueryOutput) -> Suspicion | None:
    """Whether a successful result looks wrong. None means it looks fine."""
    if not output.ok:
        return None

    if not output.rows:
        return Suspicion(
            EMPTY_RESULT,
            "The query ran but returned no rows at all. A filter is probably "
            "matching nothing — check that any text value you compared against "
            "is spelled and capitalised exactly as it appears in the data.",
        )

    single_cell = len(output.rows) == 1 and len(output.rows[0]) == 1
    if single_cell and output.rows[0][0] is None:
        return Suspicion(
            NULL_SCALAR,
            "The query returned a single NULL. An aggregate over rows that "
            "matched nothing produces this, so the filter is probably wrong.",
        )

    if expects_single_value(question) and len(output.rows) > 1:
        return Suspicion(
            EXPECTED_ONE_ROW,
            f"The question asks for a single value but the query returned "
            f"{len(output.rows)} rows. An aggregate or a missing GROUP BY is "
            f"the usual cause.",
        )

    return None
