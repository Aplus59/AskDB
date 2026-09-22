"""Degrading to another model when the preferred one is unavailable.

Free-tier Gemini capacity moves: a model that answers now returns 503 a minute
later, and the `-lite` variants are unavailable for long stretches. Waiting it
out does not work, because a demand spike outlasts any reasonable backoff.
Switching models does.

So retries stay short and this layer sits above them: fail fast on one model,
move to the next. Results answered by a fallback are marked `degraded` so they
are never mistaken for a measurement of the model that was requested.
"""

from collections.abc import Sequence
from dataclasses import replace

from google.genai import errors

from askdb.llm.client import ModelClient
from askdb.llm.retry import RetriesExhausted
from askdb.llm.types import Completion

# Worth trying a different model for. 404 means the model is retired, 429 and
# 503 mean this one is saturated. A 400 or 401 is our own bug or a bad key,
# and repeating it against every model in the chain only wastes quota.
FALLBACK_CODES = frozenset({404, 429, 503})


class NoModelAvailable(Exception):
    """Every model in the chain failed."""

    def __init__(self, models: Sequence[str], last: BaseException) -> None:
        super().__init__(f"no model answered (tried {', '.join(models)}): {last}")
        self.models = tuple(models)
        self.last = last


def should_fall_back(error: BaseException) -> bool:
    if isinstance(error, RetriesExhausted):
        return True
    if isinstance(error, errors.APIError):
        return getattr(error, "code", None) in FALLBACK_CODES
    return False


class FallbackClient:
    """Tries each model in turn until one answers."""

    def __init__(self, inner: ModelClient, chain: Sequence[str]) -> None:
        if not chain:
            raise ValueError("a fallback chain needs at least one model")
        self._inner = inner
        self._chain = tuple(chain)

    def _order(self, preferred: str | None) -> tuple[str, ...]:
        if preferred is None:
            return self._chain
        return (preferred, *(model for model in self._chain if model != preferred))

    def complete(
        self, prompt: str, *, model: str | None = None, temperature: float = 0.0
    ) -> Completion:
        order = self._order(model)
        last: BaseException | None = None

        for candidate in order:
            try:
                completion = self._inner.complete(
                    prompt, model=candidate, temperature=temperature
                )
            except Exception as error:
                if not should_fall_back(error):
                    raise
                last = error
                continue

            return replace(completion, degraded=candidate != order[0])

        assert last is not None
        raise NoModelAvailable(order, last)
