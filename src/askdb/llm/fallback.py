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

from askdb.llm.client import ModelClient, QuotaExhausted
from askdb.llm.retry import RetriesExhausted
from askdb.llm.types import Completion

# Worth trying a different model for. 404 means the model is retired, 429 and
# 503 mean this one is saturated. A 400 or 401 is our own bug or a bad key,
# and repeating it against every model in the chain only wastes quota.
FALLBACK_CODES = frozenset({404, 429, 503})


class NoModelAvailable(Exception):
    """Every model in the chain failed."""

    def __init__(self, models: Sequence[str], failures: Sequence[BaseException]) -> None:
        super().__init__(
            f"no model answered (tried {', '.join(models)}): {failures[-1]}"
        )
        self.models = tuple(models)
        self.failures = tuple(failures)
        self.last = failures[-1]

    @property
    def out_of_quota(self) -> bool:
        """Whether every model failed because its daily quota ran out.

        The distinction decides whether a long run should stop or carry on.
        Quota everywhere means the next question fails identically, so
        stopping is right. A mix of quota and transient saturation does not:
        the saturated model may answer the very next question, and halting a
        500-question run over one unlucky moment throws away the afternoon.
        """
        return all(isinstance(failure, QuotaExhausted) for failure in self.failures)


def should_fall_back(error: BaseException) -> bool:
    # Quotas are counted per model, so another model may still have room.
    if isinstance(error, QuotaExhausted | RetriesExhausted):
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
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        variant: int = 0,
    ) -> Completion:
        order = self._order(model)
        failures: list[BaseException] = []

        for candidate in order:
            try:
                completion = self._inner.complete(
                    prompt, model=candidate, temperature=temperature, variant=variant
                )
            except Exception as error:
                if not should_fall_back(error):
                    raise
                failures.append(error)
                continue

            return replace(completion, degraded=candidate != order[0])

        assert failures
        raise NoModelAvailable(order, failures)
