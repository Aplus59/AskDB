from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from google import genai
from google.genai import errors

from askdb.llm import retry
from askdb.llm.cache import ResponseCache
from askdb.llm.client import (
    GeminiClient,
    QuotaExhausted,
    is_daily_quota,
    is_transient,
    retry_after_seconds,
)

INSTANT = retry.RetryPolicy(base_delay=0.0)


class FakeUsage:
    def __init__(self, prompt: int, candidates: int) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates


class FakeResponse:
    def __init__(self, text: str | None, usage: FakeUsage | None = None) -> None:
        self.text = text
        self.usage_metadata = usage if usage is not None else FakeUsage(10, 5)


class FakeModels:
    """Stands in for `genai.Client().models`, replaying scripted outcomes."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str]] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.calls.append((model, contents))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def build(models: FakeModels, cache: ResponseCache | None = None) -> GeminiClient:
    fake = type("FakeGenai", (), {"models": models})()
    return GeminiClient(
        client=cast(genai.Client, fake),
        cache=cache,
        policy=INSTANT,
        default_model="test-model",
        sleep=lambda _: None,
    )


def test_returns_the_generated_text() -> None:
    client = build(FakeModels(FakeResponse("SELECT 1")))
    assert client.complete("hello").text == "SELECT 1"


def test_records_token_usage() -> None:
    client = build(FakeModels(FakeResponse("ok", FakeUsage(120, 8))))
    usage = client.complete("hello").usage

    assert usage.input_tokens == 120
    assert usage.output_tokens == 8
    assert usage.total_tokens == 128


def test_missing_usage_metadata_becomes_zero() -> None:
    # Some responses arrive without usage metadata; that must not crash the
    # cost accounting.
    response = FakeResponse("ok")
    response.usage_metadata = None  # type: ignore[assignment]

    assert build(FakeModels(response)).complete("hello").usage.total_tokens == 0


def test_empty_text_is_not_none() -> None:
    client = build(FakeModels(FakeResponse(None)))
    assert client.complete("hello").text == ""


def test_uses_the_default_model_when_none_is_given() -> None:
    models = FakeModels(FakeResponse("ok"))
    build(models).complete("hello")
    assert models.calls[0][0] == "test-model"


def test_an_explicit_model_overrides_the_default() -> None:
    models = FakeModels(FakeResponse("ok"))
    build(models).complete("hello", model="other-model")
    assert models.calls[0][0] == "other-model"


def test_retries_a_rate_limit_then_succeeds() -> None:
    models = FakeModels(
        errors.ClientError(429, {"error": {"message": "quota exceeded"}}),
        FakeResponse("recovered"),
    )
    result = build(models).complete("hello")

    assert result.text == "recovered"
    assert result.attempts == 2
    assert len(models.calls) == 2


def test_retries_a_server_error() -> None:
    models = FakeModels(
        errors.ServerError(503, {"error": {"message": "unavailable"}}),
        FakeResponse("recovered"),
    )
    assert build(models).complete("hello").text == "recovered"


def test_retries_a_dns_failure() -> None:
    # A real 110-question run lost 53 questions to `getaddrinfo failed`
    # because connection errors were not classified as retryable.
    models = FakeModels(
        httpx.ConnectError("[Errno 11001] getaddrinfo failed"),
        FakeResponse("recovered"),
    )
    result = build(models).complete("hello")

    assert result.text == "recovered"
    assert result.attempts == 2


def test_retries_a_read_timeout() -> None:
    models = FakeModels(httpx.ReadTimeout("timed out"), FakeResponse("recovered"))
    assert build(models).complete("hello").text == "recovered"


def test_a_bad_request_is_not_retried() -> None:
    # Retrying a malformed request burns quota and cannot succeed.
    models = FakeModels(errors.ClientError(400, {"error": {"message": "bad request"}}))

    with pytest.raises(errors.APIError):
        build(models).complete("hello")

    assert len(models.calls) == 1


def test_persistent_rate_limiting_eventually_gives_up() -> None:
    models = FakeModels(*[errors.ClientError(429, {}) for _ in range(5)])

    with pytest.raises(retry.RetriesExhausted):
        build(models).complete("hello")


def test_a_cache_hit_makes_no_api_call(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    models = FakeModels(FakeResponse("first"))

    first = build(models, cache).complete("hello")
    second = build(FakeModels(), cache).complete("hello")

    assert first.text == second.text == "first"
    assert not first.cached
    assert second.cached


def test_a_cache_miss_is_stored(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    build(FakeModels(FakeResponse("stored")), cache).complete("hello")
    assert len(cache) == 1


def test_different_temperatures_are_cached_separately(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    models = FakeModels(FakeResponse("cold"), FakeResponse("warm"))
    client = build(models, cache)

    assert client.complete("hello", temperature=0.0).text == "cold"
    assert client.complete("hello", temperature=1.0).text == "warm"
    assert len(cache) == 2


def test_repeated_draws_are_cached_separately(tmp_path: Path) -> None:
    # Without this, sampling the same prompt five times returns five copies of
    # one cached response, and measuring agreement between them would be
    # measuring the cache.
    cache = ResponseCache(tmp_path / "c.sqlite")
    models = FakeModels(FakeResponse("first"), FakeResponse("second"))
    client = build(models, cache)

    assert client.complete("hello", variant=0).text == "first"
    assert client.complete("hello", variant=1).text == "second"
    assert len(cache) == 2


def test_the_same_variant_still_hits_the_cache(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    models = FakeModels(FakeResponse("first"))
    client = build(models, cache)

    assert client.complete("hello", variant=3).text == "first"
    assert client.complete("hello", variant=3).cached


@pytest.mark.parametrize("code", [408, 429])
def test_transient_client_codes(code: int) -> None:
    assert is_transient(errors.ClientError(code, {}))


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_permanent_client_codes(code: int) -> None:
    assert not is_transient(errors.ClientError(code, {}))


@pytest.mark.parametrize("code", [500, 503])
def test_server_errors_are_always_transient(code: int) -> None:
    assert is_transient(errors.ServerError(code, {}))


def quota_payload(quota_id: str, retry_delay: str | None = None) -> dict[str, Any]:
    details: list[dict[str, Any]] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id}],
        }
    ]
    if retry_delay is not None:
        details.append(
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
        )
    return {"error": {"code": 429, "message": "quota", "details": details}}


def test_a_daily_quota_is_not_retried() -> None:
    # A per-minute limit clears in seconds. A daily quota clears at midnight,
    # so thirty seconds of backoff is thirty seconds wasted.
    payload = quota_payload("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "35s")
    models = FakeModels(errors.ClientError(429, payload), FakeResponse("never reached"))

    with pytest.raises(QuotaExhausted) as caught:
        build(models).complete("hello")

    assert caught.value.model == "test-model"
    assert len(models.calls) == 1


def test_a_per_minute_rate_limit_is_still_retried() -> None:
    payload = quota_payload("GenerateRequestsPerMinutePerProject-FreeTier", "5s")
    models = FakeModels(errors.ClientError(429, payload), FakeResponse("recovered"))

    assert build(models).complete("hello").text == "recovered"


def test_daily_quota_detection() -> None:
    daily = errors.ClientError(429, quota_payload("GenerateRequestsPerDayPerModel"))
    minute = errors.ClientError(429, quota_payload("RequestsPerMinutePerProject"))

    assert is_daily_quota(daily)
    assert not is_daily_quota(minute)
    assert not is_daily_quota(errors.ClientError(429, {}))


def test_reads_the_servers_retry_hint() -> None:
    error = errors.ClientError(429, quota_payload("PerMinute", "35.8s"))
    assert retry_after_seconds(error) == pytest.approx(35.8)


def test_a_missing_or_malformed_hint_is_ignored() -> None:
    assert retry_after_seconds(errors.ClientError(429, quota_payload("PerMinute"))) is None
    assert (
        retry_after_seconds(errors.ClientError(429, quota_payload("PerMinute", "soon")))
        is None
    )
