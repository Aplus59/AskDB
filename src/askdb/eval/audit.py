"""Classifying why an answer was scored wrong.

A benchmark score is not evidence on its own. Published analysis found
annotation error rates above 50% on BIRD Mini-Dev, with corrections moving
leaderboard positions by up to nine places. So "we scored X" has to be
accompanied by "and here is what the failures actually were".

Labels are a human judgement recorded here, not something inferred. The value
of the exercise comes from reading the queries and their results; a classifier
would only reproduce the model's own blind spots.

Categories
----------
model_error      The query is genuinely wrong: bad join, wrong aggregation,
                 wrong filter, missed condition.
format_mismatch  The right data, shaped differently. Returning "201307" where
                 the reference returns "07", an extra column, a different
                 rounding. The reasoning was correct.
ambiguous        The question admits more than one defensible reading and the
                 reference picked a different one.
annotation_error The reference query does not answer the question as asked.
schema_limit     The question cannot be answered from this schema at all.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from askdb.eval.run import QuestionResult

MODEL_ERROR = "model_error"
FORMAT_MISMATCH = "format_mismatch"
AMBIGUOUS = "ambiguous"
ANNOTATION_ERROR = "annotation_error"
SCHEMA_LIMIT = "schema_limit"

CATEGORIES = (MODEL_ERROR, FORMAT_MISMATCH, AMBIGUOUS, ANNOTATION_ERROR, SCHEMA_LIMIT)

# Questions the benchmark cannot score fairly. Excluded from the denominator
# of corrected accuracy rather than counted as successes: claiming credit for
# them would overstate the system as badly as the raw score understates it.
UNSCORABLE = frozenset({ANNOTATION_ERROR, SCHEMA_LIMIT})


class UnknownCategory(ValueError):
    pass


@dataclass(frozen=True)
class Label:
    question_id: int
    category: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise UnknownCategory(
                f"{self.category!r} is not one of: {', '.join(CATEGORIES)}"
            )


def save_label(path: Path, label: Label) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")


def load_labels(path: Path) -> dict[int, Label]:
    """Read labels back, with later entries overriding earlier ones.

    Re-labelling is expected: a question read again after seeing its result
    rows often looks different.
    """
    if not path.is_file():
        return {}

    found: dict[int, Label] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            label = Label(**data)
        except (json.JSONDecodeError, TypeError, UnknownCategory):
            continue
        found[label.question_id] = label
    return found


@dataclass(frozen=True)
class AuditSummary:
    total: int
    matched: int
    failures: int
    labelled: int
    counts: dict[str, int]

    @property
    def unlabelled(self) -> int:
        return self.failures - self.labelled

    @property
    def raw_accuracy(self) -> float:
        return self.matched / self.total if self.total else 0.0

    @property
    def corrected_accuracy(self) -> float:
        """Accuracy over the questions the benchmark can score fairly.

        Questions labelled as annotation errors or unanswerable are removed
        from the denominator entirely.
        """
        excluded = sum(self.counts.get(category, 0) for category in UNSCORABLE)
        scorable = self.total - excluded
        return self.matched / scorable if scorable else 0.0

    @property
    def semantic_accuracy(self) -> float:
        """Raw accuracy plus failures that found the right data in a different shape.

        Reported alongside, never instead: a system whose output shape does
        not match what was asked for is still failing the user, even when the
        reasoning behind it was sound.
        """
        if not self.total:
            return 0.0
        return (self.matched + self.counts.get(FORMAT_MISMATCH, 0)) / self.total

    def render(self) -> str:
        lines = [
            f"questions           {self.total}",
            f"correct             {self.matched}",
            f"failures            {self.failures}  ({self.labelled} labelled,"
            f" {self.unlabelled} unlabelled)",
            "",
            f"raw accuracy        {self.raw_accuracy:.3f}",
            f"corrected accuracy  {self.corrected_accuracy:.3f}"
            "   (excluding unscorable questions)",
            f"semantically right  {self.semantic_accuracy:.3f}"
            "   (including shape mismatches)",
        ]
        if self.counts:
            lines.append("")
            lines.append("failure taxonomy:")
            for category in CATEGORIES:
                count = self.counts.get(category, 0)
                if count:
                    share = count / self.failures if self.failures else 0.0
                    lines.append(f"  {category:<18} {count:>3}  ({share:.0%} of failures)")
        if self.unlabelled:
            lines.append("")
            lines.append(f"{self.unlabelled} failures still unlabelled")
        return "\n".join(lines)


def summarize(results: Sequence[QuestionResult], labels: dict[int, Label]) -> AuditSummary:
    failures = [result for result in results if not result.match]
    counts: dict[str, int] = {}
    labelled = 0

    for failure in failures:
        label = labels.get(failure.question_id)
        if label is None:
            continue
        labelled += 1
        counts[label.category] = counts.get(label.category, 0) + 1

    return AuditSummary(
        total=len(results),
        matched=sum(1 for result in results if result.match),
        failures=len(failures),
        labelled=labelled,
        counts=counts,
    )
