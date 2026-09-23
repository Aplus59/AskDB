"""Turning token counts into money.

Every cost in this project is measured in tokens, which is the right unit for
engineering and the wrong one for a decision. "1,357 tokens per question" does
not tell anyone whether to ship something; "$4 per thousand questions" does.

Prices are hard-coded with a date, and a model that is not in the table is
reported as unpriced rather than estimated. A plausible-looking cost derived
from a guessed rate is worse than an honest gap, because nobody checks a
number that looks reasonable.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from askdb.eval.run import QuestionResult

# USD per million tokens, (input, output). Google AI Studio paid tier.
# Checked 2026-09-23 — verify before quoting these anywhere that matters.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}

PRICES_CHECKED = "2026-09-23"
PER_MILLION = 1_000_000


@dataclass(frozen=True)
class CostReport:
    priced_questions: int
    unpriced_questions: int
    assumed_questions: int
    input_tokens: int
    output_tokens: int
    usd: float
    unpriced_models: tuple[str, ...]
    assumed_model: str | None = None

    @property
    def total_questions(self) -> int:
        return self.priced_questions + self.unpriced_questions

    @property
    def usd_per_question(self) -> float:
        return self.usd / self.priced_questions if self.priced_questions else 0.0

    @property
    def usd_per_thousand(self) -> float:
        return self.usd_per_question * 1000

    def render(self) -> str:
        lines = [
            f"priced questions   {self.priced_questions}",
            f"input tokens       {self.input_tokens:,}",
            f"output tokens      {self.output_tokens:,}",
            f"total              ${self.usd:.4f}",
            f"per question       ${self.usd_per_question:.6f}",
            f"per 1000 questions ${self.usd_per_thousand:.2f}",
            "",
            f"prices checked {PRICES_CHECKED}; verify before quoting.",
        ]
        if self.assumed_questions:
            lines.insert(
                1,
                f"assumed {self.assumed_model}  for {self.assumed_questions} question(s) "
                "that did not record which model answered",
            )
        if self.unpriced_questions:
            lines.insert(
                1,
                f"unpriced questions {self.unpriced_questions}"
                f"  ({', '.join(self.unpriced_models)})",
            )
        return "\n".join(lines)


def price_of(model: str | None) -> tuple[float, float] | None:
    return PRICES.get(model) if model else None


def estimate(
    results: Sequence[QuestionResult], assume_model: str | None = None
) -> CostReport:
    """Total the cost of a run.

    `assume_model` prices questions that did not record which model answered —
    runs made before that field existed. It is opt-in and named in the output,
    so the assumption travels with the number instead of disappearing into it.
    """
    priced = 0
    unpriced = 0
    assumed = 0
    unpriced_models: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    usd = 0.0

    for result in results:
        model = result.model or assume_model
        if result.model is None and assume_model is not None:
            assumed += 1

        price = price_of(model)
        if price is None:
            unpriced += 1
            unpriced_models.add(model or "unknown")
            continue

        per_input, per_output = price
        priced += 1
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        usd += (
            result.input_tokens * per_input + result.output_tokens * per_output
        ) / PER_MILLION

    return CostReport(
        priced_questions=priced,
        unpriced_questions=unpriced,
        assumed_questions=assumed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=usd,
        unpriced_models=tuple(sorted(unpriced_models)),
        assumed_model=assume_model if assumed else None,
    )
