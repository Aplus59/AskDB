"""Running the agent over a dataset and scoring what it produced.

Results are written one JSON object per line as they are produced, rather than
collected and saved at the end. A full Mini-Dev run is 500 questions against a
rate-limited API and takes hours, so it will be interrupted — by a quota
ceiling, a dropped connection, or a closed laptop. Appending as it goes means
an interrupted run is resumable instead of lost.

Per-question records are kept, not just the aggregate. The failure audit reads
them back, and an average cannot be re-examined.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from askdb.agent.loop import Agent
from askdb.data.minidev import Question
from askdb.eval import execution


@dataclass(frozen=True)
class QuestionResult:
    question_id: int
    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str | None
    match: bool
    exact_match: bool
    failure_reason: str | None
    repairs: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tables_shown: tuple[str, ...]
    difficulty: str | None = None
    rechecks: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def recovered(self) -> bool:
        """Correct only because a repair fixed it."""
        return self.match and self.repairs > 0

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["tables_shown"] = list(self.tables_shown)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "QuestionResult":
        payload = dict(data)
        payload["tables_shown"] = tuple(payload.get("tables_shown", ()))
        return cls(**payload)


def run_question(agent: Agent, question: Question, database: Path) -> QuestionResult:
    result = agent.answer(question.question, question.evidence)
    predicted = result.sql

    match = False
    exact = False
    reason: str | None = "no_sql"

    if predicted:
        comparison = execution.score(database, predicted, question.gold_sql)
        match = comparison.match
        exact = comparison.exact_match
        reason = comparison.failure_reason

    return QuestionResult(
        question_id=question.question_id,
        db_id=question.db_id,
        question=question.question,
        gold_sql=question.gold_sql,
        predicted_sql=predicted,
        match=match,
        exact_match=exact,
        failure_reason=reason,
        repairs=result.repairs,
        model_calls=result.model_calls,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        latency_ms=sum(step.latency_ms for step in result.steps),
        tables_shown=result.tables_shown,
        difficulty=question.difficulty,
        rechecks=result.rechecks,
    )


def append_result(path: Path, result: QuestionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_json(), ensure_ascii=False) + "\n")


def load_results(path: Path) -> dict[int, QuestionResult]:
    """Read back a previous run, keyed by question id.

    Malformed trailing lines are ignored: a run killed mid-write leaves a
    partial line, and refusing to load because of it would throw away every
    completed question.
    """
    if not path.is_file():
        return {}

    found: dict[int, QuestionResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            result = QuestionResult.from_json(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
        found[result.question_id] = result
    return found


@dataclass(frozen=True)
class Summary:
    total: int
    accuracy: float
    strict_accuracy: float
    repair_rate: float
    recovery_rate: float
    recheck_rate: float
    avg_tokens: float
    avg_model_calls: float
    failures: dict[str, int]
    by_difficulty: dict[str, tuple[int, float]]

    def render(self) -> str:
        lines = [
            f"questions          {self.total}",
            f"execution accuracy {self.accuracy:.3f}",
            f"strict accuracy    {self.strict_accuracy:.3f}",
            f"needed a repair    {self.repair_rate:.3f}",
            f"saved by a repair  {self.recovery_rate:.3f}",
            f"self-check fired   {self.recheck_rate:.3f}",
            f"tokens / question  {self.avg_tokens:.0f}",
            f"calls / question   {self.avg_model_calls:.2f}",
        ]
        if self.by_difficulty:
            lines.append("")
            lines.append("by difficulty:")
            for name, (count, accuracy) in sorted(self.by_difficulty.items()):
                lines.append(f"  {name:<10} {accuracy:.3f}  (n={count})")
        if self.failures:
            lines.append("")
            lines.append("failures:")
            for reason, count in sorted(self.failures.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {reason:<18} {count}")
        return "\n".join(lines)


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize(results: Sequence[QuestionResult]) -> Summary:
    total = len(results)
    if total == 0:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {})

    failures: dict[str, int] = {}
    for result in results:
        if not result.match and result.failure_reason:
            failures[result.failure_reason] = failures.get(result.failure_reason, 0) + 1

    by_difficulty: dict[str, tuple[int, float]] = {}
    levels = {result.difficulty for result in results if result.difficulty}
    for level in levels:
        group = [result for result in results if result.difficulty == level]
        by_difficulty[level] = (
            len(group),
            _ratio(sum(1 for r in group if r.match), len(group)),
        )

    return Summary(
        total=total,
        accuracy=_ratio(sum(1 for r in results if r.match), total),
        strict_accuracy=_ratio(sum(1 for r in results if r.exact_match), total),
        repair_rate=_ratio(sum(1 for r in results if r.repairs > 0), total),
        recovery_rate=_ratio(sum(1 for r in results if r.recovered), total),
        recheck_rate=_ratio(sum(1 for r in results if r.rechecks > 0), total),
        avg_tokens=sum(r.total_tokens for r in results) / total,
        avg_model_calls=sum(r.model_calls for r in results) / total,
        failures=failures,
        by_difficulty=by_difficulty,
    )


def pending(questions: Iterable[Question], done: dict[int, QuestionResult]) -> list[Question]:
    return [question for question in questions if question.question_id not in done]


def stratified_sample(questions: Sequence[Question], limit: int) -> list[Question]:
    """Take `limit` questions spread across databases.

    Mini-Dev is ordered by database, so taking the first N draws them all from
    one schema. A number measured that way describes one database, not the
    benchmark, and will not survive contact with the full run.

    Round-robin rather than random: the same subset every time means two runs
    are comparable.
    """
    if limit <= 0:
        return []

    by_database: dict[str, list[Question]] = {}
    for question in questions:
        by_database.setdefault(question.db_id, []).append(question)

    chosen: list[Question] = []
    depth = 0
    while len(chosen) < limit:
        added = False
        for db_id in sorted(by_database):
            group = by_database[db_id]
            if depth < len(group):
                chosen.append(group[depth])
                added = True
                if len(chosen) == limit:
                    return chosen
        if not added:
            break
        depth += 1
    return chosen
