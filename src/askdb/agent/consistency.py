"""Measuring how sure the model is, by asking it more than once.

The abstention signals available so far — repairs used, an unresolved concern,
vocabulary coverage — are all rare, so the confidence score was effectively
binary and produced no usable curve.

Self-consistency gives a continuous one. Draw the query several times at a
non-zero temperature and see how much the answers agree. A question the model
finds easy produces the same result every time; one it is guessing at
scatters. The share of draws landing on the most common answer is a number
between zero and one, which is exactly what the thresholds needed.

Two decisions matter here:

- **Agreement is measured on results, not on SQL text.** Two queries can be
  written completely differently and return identical rows; counting those as
  disagreement would report confusion where there is none.
- **Failed queries vote too, as their own group.** A draw that does not
  execute is evidence of uncertainty, and dropping it would flatter the score
  of a question the model kept getting wrong.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from askdb.agent.tools import QueryOutput
from askdb.eval.execution import row_signature

FAILED = "__failed__"


@dataclass(frozen=True)
class Draw:
    sql: str
    output: QueryOutput


@dataclass(frozen=True)
class Consensus:
    sql: str
    output: QueryOutput
    agreement: float
    draws: int
    distinct_answers: int

    @property
    def unanimous(self) -> bool:
        return self.distinct_answers == 1


def signature(output: QueryOutput) -> Any:
    """What counts as "the same answer" for voting purposes."""
    if not output.ok:
        return FAILED
    return row_signature(output.rows)


def vote(draws: Sequence[Draw]) -> Consensus:
    """Pick the most agreed-upon answer and score the agreement.

    Ties go to the earliest draw, so the result does not depend on dictionary
    ordering and two runs over the same draws agree with each other.
    """
    if not draws:
        raise ValueError("cannot vote on nothing")

    signatures = [signature(draw.output) for draw in draws]
    counts = Counter(signatures)

    best_count = max(counts.values())
    winning = next(sig for sig in signatures if counts[sig] == best_count)
    winner = next(
        draw for draw, sig in zip(draws, signatures, strict=True) if sig == winning
    )

    # A group of failures winning means the model could not produce a working
    # query at all. Prefer any draw that actually executed, while still
    # reporting the low agreement that made this ambiguous.
    if winning == FAILED:
        succeeded = [draw for draw, sig in zip(draws, signatures, strict=True) if sig != FAILED]
        if succeeded:
            winner = succeeded[0]

    return Consensus(
        sql=winner.sql,
        output=winner.output,
        agreement=best_count / len(draws),
        draws=len(draws),
        distinct_answers=len(counts),
    )
