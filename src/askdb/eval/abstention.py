"""Sweeping the abstention thresholds.

The confidence score depends only on signals already recorded per question:
repairs used, whether a concern survived a recheck, and vocabulary coverage.
Nothing about it needs the model. So the whole sweep is recomputed offline
from a finished run — every operating point, in milliseconds, at no cost.

That is worth stating plainly, because the usual way to produce a curve like
this is to re-run the system once per threshold. Here the expensive part
(answering the questions) happens once, and the cheap part (deciding whether
to stand behind each answer) is replayed as often as needed.

The number to watch is the silent-error rate: answers given confidently that
were wrong. Coverage can always be raised by answering more, and accuracy can
always be raised by answering less. Only the pair of them together describes
the system.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from askdb.agent import confidence
from askdb.eval.run import QuestionResult

FAILED_REASONS = frozenset({"execution_error", "timeout", "no_sql", "gold_failed"})


def decide(
    result: QuestionResult, clarify_below: float, refuse_below: float
) -> str:
    """Recompute the decision for one question at given thresholds."""
    if not result.match and result.failure_reason in FAILED_REASONS:
        return confidence.REFUSE

    score = 1.0
    if result.agreement is not None and result.agreement < 1.0:
        score -= (1.0 - result.agreement) * confidence.DISAGREEMENT_WEIGHT
    if result.unresolved_concern:
        score -= confidence.UNRESOLVED_CONCERN_PENALTY
    if result.repairs:
        score -= confidence.REPAIR_PENALTY * result.repairs
    if result.coverage < confidence.COVERAGE_FLOOR:
        score -= confidence.LOW_COVERAGE_PENALTY
    score = max(0.0, min(1.0, score))

    if score < refuse_below:
        return confidence.REFUSE
    if score < clarify_below:
        return confidence.CLARIFY
    return confidence.ANSWER


@dataclass(frozen=True)
class OperatingPoint:
    clarify_below: float
    refuse_below: float
    total: int
    answered: int
    correct: int
    silent_errors: int
    abstained_wrong: int

    @property
    def coverage(self) -> float:
        """Share of questions the system was willing to answer."""
        return self.answered / self.total if self.total else 0.0

    @property
    def accuracy_when_answered(self) -> float:
        return self.correct / self.answered if self.answered else 0.0

    @property
    def silent_error_rate(self) -> float:
        """Share of all questions answered confidently and wrongly.

        The number a business actually cares about, and the one almost no
        text-to-SQL project reports.
        """
        return self.silent_errors / self.total if self.total else 0.0

    @property
    def abstained(self) -> int:
        return self.total - self.answered

    @property
    def abstention_precision(self) -> float:
        """Of the questions declined, how many would have been wrong.

        Low precision means the system is refusing work it could have done.
        """
        return self.abstained_wrong / self.abstained if self.abstained else 0.0


def evaluate_at(
    results: Sequence[QuestionResult], clarify_below: float, refuse_below: float
) -> OperatingPoint:
    answered = 0
    correct = 0
    silent_errors = 0
    abstained_wrong = 0

    for result in results:
        if decide(result, clarify_below, refuse_below) == confidence.ANSWER:
            answered += 1
            if result.match:
                correct += 1
            else:
                silent_errors += 1
        elif not result.match:
            abstained_wrong += 1

    return OperatingPoint(
        clarify_below=clarify_below,
        refuse_below=refuse_below,
        total=len(results),
        answered=answered,
        correct=correct,
        silent_errors=silent_errors,
        abstained_wrong=abstained_wrong,
    )


def sweep(
    results: Sequence[QuestionResult],
    thresholds: Sequence[float] = (0.0, 0.3, 0.45, 0.6, 0.75, 0.85, 0.95, 1.01),
    refuse_below: float = confidence.DEFAULT_REFUSE_BELOW,
) -> list[OperatingPoint]:
    return [evaluate_at(results, threshold, refuse_below) for threshold in thresholds]


def render(points: Sequence[OperatingPoint]) -> str:
    header = (
        f"{'clarify<':>9}  {'coverage':>8}  {'acc|ans':>8}  "
        f"{'silent err':>10}  {'abst prec':>9}  {'answered':>8}"
    )
    lines = [header, "-" * len(header)]
    for point in points:
        lines.append(
            f"{point.clarify_below:>9.2f}  "
            f"{point.coverage:>8.3f}  "
            f"{point.accuracy_when_answered:>8.3f}  "
            f"{point.silent_error_rate:>10.3f}  "
            f"{point.abstention_precision:>9.3f}  "
            f"{point.answered:>4}/{point.total:<3}"
        )
    return "\n".join(lines)
