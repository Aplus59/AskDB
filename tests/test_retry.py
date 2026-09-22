import pytest

from askdb.llm import retry


def test_returns_immediately_when_the_call_succeeds() -> None:
    calls = []

    def call() -> str:
        calls.append(1)
        return "ok"

    assert retry.with_retries(call, sleep=lambda _: None) == "ok"
    assert len(calls) == 1


def test_retries_until_it_succeeds() -> None:
    attempts = []

    def call() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise retry.TransientError("rate limited")
        return "ok"

    waits: list[float] = []
    result = retry.with_retries(call, sleep=waits.append, uniform=lambda _, high: high)

    assert result == "ok"
    assert len(attempts) == 3
    assert waits == [1.0, 2.0]


def test_gives_up_after_the_attempt_limit() -> None:
    policy = retry.RetryPolicy(max_attempts=3)

    def call() -> str:
        raise retry.TransientError("still rate limited")

    with pytest.raises(retry.RetriesExhausted) as caught:
        retry.with_retries(call, policy, sleep=lambda _: None)

    assert caught.value.attempts == 3
    assert isinstance(caught.value.last, retry.TransientError)


def test_sleeps_one_fewer_time_than_it_attempts() -> None:
    # No point waiting after the final failure; nothing follows it.
    policy = retry.RetryPolicy(max_attempts=4)
    waits: list[float] = []

    def call() -> str:
        raise retry.TransientError("nope")

    with pytest.raises(retry.RetriesExhausted):
        retry.with_retries(call, policy, sleep=waits.append, uniform=lambda _, high: high)

    assert len(waits) == 3


def test_other_exceptions_are_not_retried() -> None:
    attempts = []

    def call() -> str:
        attempts.append(1)
        raise ValueError("a bug, not a rate limit")

    with pytest.raises(ValueError):
        retry.with_retries(call, sleep=lambda _: None)

    assert len(attempts) == 1


def test_backoff_is_exponential() -> None:
    policy = retry.RetryPolicy(base_delay=2.0, max_delay=100.0)
    assert [policy.ceiling_for(n) for n in (1, 2, 3, 4)] == [2.0, 4.0, 8.0, 16.0]


def test_backoff_is_capped() -> None:
    policy = retry.RetryPolicy(base_delay=1.0, max_delay=5.0)
    assert policy.ceiling_for(10) == 5.0


def test_attempt_numbering_is_one_based() -> None:
    with pytest.raises(ValueError, match="1-based"):
        retry.RetryPolicy().ceiling_for(0)


def test_jitter_draws_between_zero_and_the_ceiling() -> None:
    policy = retry.RetryPolicy(base_delay=4.0)
    bounds: list[tuple[float, float]] = []

    def call() -> str:
        raise retry.TransientError("nope")

    def record(low: float, high: float) -> float:
        bounds.append((low, high))
        return low

    with pytest.raises(retry.RetriesExhausted):
        retry.with_retries(
            call,
            retry.RetryPolicy(max_attempts=2, base_delay=4.0),
            sleep=lambda _: None,
            uniform=record,
        )

    assert bounds == [(0.0, policy.ceiling_for(1))]
