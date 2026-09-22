import pytest
from google.genai import errors

from askdb.llm.fallback import FallbackClient, NoModelAvailable, should_fall_back
from askdb.llm.retry import RetriesExhausted, TransientError
from askdb.llm.types import Completion, Usage


class ScriptedClient:
    """A model client whose behaviour is declared per model name."""

    def __init__(self, outcomes: dict[str, object]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def complete(
        self, prompt: str, *, model: str | None = None, temperature: float = 0.0
    ) -> Completion:
        assert model is not None
        self.calls.append(model)
        outcome = self.outcomes.get(model, errors.ClientError(404, {}))
        if isinstance(outcome, Exception):
            raise outcome
        return Completion(text=str(outcome), model=model, usage=Usage(1, 1))


def exhausted() -> RetriesExhausted:
    return RetriesExhausted(5, TransientError("503 UNAVAILABLE"))


def test_uses_the_first_model_when_it_answers() -> None:
    inner = ScriptedClient({"a": "from a", "b": "from b"})
    result = FallbackClient(inner, ["a", "b"]).complete("q")

    assert result.text == "from a"
    assert result.model == "a"
    assert not result.degraded
    assert inner.calls == ["a"]


def test_falls_back_when_the_first_is_saturated() -> None:
    inner = ScriptedClient({"a": exhausted(), "b": "from b"})
    result = FallbackClient(inner, ["a", "b"]).complete("q")

    assert result.text == "from b"
    assert result.model == "b"
    assert inner.calls == ["a", "b"]


def test_a_fallback_result_is_marked_degraded() -> None:
    inner = ScriptedClient({"a": exhausted(), "b": "from b"})
    assert FallbackClient(inner, ["a", "b"]).complete("q").degraded


def test_falls_back_past_a_retired_model() -> None:
    inner = ScriptedClient({"a": errors.ClientError(404, {}), "b": "from b"})
    assert FallbackClient(inner, ["a", "b"]).complete("q").text == "from b"


def test_falls_back_on_rate_limiting() -> None:
    inner = ScriptedClient({"a": errors.ClientError(429, {}), "b": "from b"})
    assert FallbackClient(inner, ["a", "b"]).complete("q").text == "from b"


def test_walks_the_whole_chain() -> None:
    inner = ScriptedClient({"a": exhausted(), "b": exhausted(), "c": "from c"})
    result = FallbackClient(inner, ["a", "b", "c"]).complete("q")

    assert result.text == "from c"
    assert inner.calls == ["a", "b", "c"]


def test_raises_when_every_model_fails() -> None:
    inner = ScriptedClient({"a": exhausted(), "b": exhausted()})

    with pytest.raises(NoModelAvailable) as caught:
        FallbackClient(inner, ["a", "b"]).complete("q")

    assert caught.value.models == ("a", "b")


def test_a_bad_request_stops_immediately() -> None:
    # Our own malformed request will fail on every model, so trying the rest
    # of the chain only wastes quota.
    inner = ScriptedClient({"a": errors.ClientError(400, {}), "b": "from b"})

    with pytest.raises(errors.APIError):
        FallbackClient(inner, ["a", "b"]).complete("q")

    assert inner.calls == ["a"]


def test_an_auth_failure_stops_immediately() -> None:
    inner = ScriptedClient({"a": errors.ClientError(401, {}), "b": "from b"})

    with pytest.raises(errors.APIError):
        FallbackClient(inner, ["a", "b"]).complete("q")

    assert inner.calls == ["a"]


def test_an_unrelated_exception_is_not_swallowed() -> None:
    inner = ScriptedClient({"a": ValueError("a bug")})

    with pytest.raises(ValueError):
        FallbackClient(inner, ["a", "b"]).complete("q")


def test_an_explicit_model_is_tried_first() -> None:
    inner = ScriptedClient({"a": "from a", "b": "from b"})
    result = FallbackClient(inner, ["a", "b"]).complete("q", model="b")

    assert result.model == "b"
    assert not result.degraded
    assert inner.calls == ["b"]


def test_an_explicit_model_still_falls_back_to_the_chain() -> None:
    inner = ScriptedClient({"a": "from a", "b": exhausted()})
    result = FallbackClient(inner, ["a", "b"]).complete("q", model="b")

    assert result.model == "a"
    assert result.degraded
    assert inner.calls == ["b", "a"]


def test_an_explicit_model_is_not_retried_later_in_the_chain() -> None:
    inner = ScriptedClient({"a": exhausted(), "b": exhausted()})

    with pytest.raises(NoModelAvailable):
        FallbackClient(inner, ["a", "b"]).complete("q", model="a")

    assert inner.calls == ["a", "b"]


def test_an_empty_chain_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        FallbackClient(ScriptedClient({}), [])


def test_classification_of_errors() -> None:
    assert should_fall_back(exhausted())
    assert should_fall_back(errors.ClientError(404, {}))
    assert should_fall_back(errors.ServerError(503, {}))
    assert not should_fall_back(errors.ClientError(400, {}))
    assert not should_fall_back(ValueError("a bug"))
