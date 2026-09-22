"""Retrying transient model failures.

The free Gemini tier allows 10-15 requests per minute, so rate limiting is the
normal case during development rather than an exceptional one. Retry behaviour
here is real logic with its own tests, not a decorator bolted on at the end.

Delays use full jitter: each wait is drawn uniformly from zero up to the
exponential ceiling, instead of being exactly the ceiling. Fixed backoff makes
every blocked caller retry at the same instant, which reproduces the overload
that caused the failure. Jitter spreads them out.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass


class TransientError(Exception):
    """A failure worth retrying, such as a rate limit or a server error."""


class RetriesExhausted(Exception):
    """Raised when every attempt failed."""

    def __init__(self, attempts: int, last: BaseException) -> None:
        super().__init__(f"giving up after {attempts} attempts: {last}")
        self.attempts = attempts
        self.last = last


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0

    def ceiling_for(self, attempt: int) -> float:
        """The longest this attempt is allowed to wait, before jitter.

        `attempt` is 1-based, so the first retry waits at most `base_delay`.
        """
        if attempt < 1:
            raise ValueError("attempt is 1-based")
        exponential = self.base_delay * (2.0 ** (attempt - 1))
        return min(exponential, self.max_delay)


def with_retries[T](
    call: Callable[[], T],
    policy: RetryPolicy | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    uniform: Callable[[float, float], float] = random.uniform,
) -> T:
    """Call something, retrying transient failures with jittered backoff.

    `sleep` and `uniform` are injectable so the policy can be tested without
    waiting and without flakiness.
    """
    policy = policy or RetryPolicy()
    last: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return call()
        except TransientError as error:
            last = error
            if attempt == policy.max_attempts:
                break
            sleep(uniform(0.0, policy.ceiling_for(attempt)))

    assert last is not None
    raise RetriesExhausted(policy.max_attempts, last)
