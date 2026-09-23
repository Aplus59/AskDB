"""Calling Gemini, with caching and retries wrapped around it.

Every call goes through the same path: look in the cache, otherwise call the
API under a retry policy, then record the result. That ordering matters —
a cached response costs nothing and cannot be rate limited, so the cache is
checked before anything else.
"""

import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from google import genai
from google.genai import errors, types

from askdb.config import settings
from askdb.llm import retry
from askdb.llm.cache import ResponseCache, cache_key
from askdb.llm.types import Completion, Usage

# 429 is the free tier's rate limit; 408 is a timeout. Both are worth another
# attempt. Every other 4xx means the request itself is wrong, and retrying an
# invalid request just wastes quota.
TRANSIENT_CLIENT_CODES = frozenset({408, 429})

# The API never answered at all: DNS failed, the connection dropped, the read
# timed out. A 110-question run lost 53 questions to `getaddrinfo failed`
# because these were not treated as retryable, so they are now.
TRANSIENT_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)


class ModelClient(Protocol):
    """What the rest of the codebase needs from a model.

    Narrow on purpose: the agent depends on this rather than on the Gemini
    SDK, so its tests run without network access or an API key.
    """

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        variant: int = 0,
    ) -> Completion: ...


class QuotaExhausted(Exception):
    """A per-day quota has run out.

    Distinct from a rate limit on purpose. A per-minute limit clears in
    seconds, so backing off works. A daily quota clears at midnight, so
    retrying is thirty wasted seconds followed by the same failure — and every
    remaining question in a run will fail the same way. The caller needs to
    know the difference so it can stop rather than grind.
    """

    def __init__(self, model: str, message: str) -> None:
        super().__init__(f"{model}: {message}")
        self.model = model


def _error_details(error: errors.APIError) -> list[dict[str, Any]]:
    payload = getattr(error, "details", None)
    if not isinstance(payload, dict):
        return []
    inner = payload.get("error")
    details = inner.get("details") if isinstance(inner, dict) else None
    return [item for item in details or [] if isinstance(item, dict)]


def is_daily_quota(error: errors.APIError) -> bool:
    """Whether a 429 is a daily quota rather than a per-minute rate limit."""
    for item in _error_details(error):
        for violation in item.get("violations") or []:
            if "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


def retry_after_seconds(error: errors.APIError) -> float | None:
    """How long the server asked us to wait, if it said."""
    for item in _error_details(error):
        raw = item.get("retryDelay")
        if isinstance(raw, str) and raw.endswith("s"):
            try:
                return float(raw[:-1])
            except ValueError:
                return None
    return None


def is_transient(error: errors.APIError) -> bool:
    if isinstance(error, errors.ServerError):
        return True
    return getattr(error, "code", None) in TRANSIENT_CLIENT_CODES


def _usage_from(response: types.GenerateContentResponse) -> Usage:
    metadata = response.usage_metadata
    if metadata is None:
        return Usage()
    return Usage(
        input_tokens=metadata.prompt_token_count or 0,
        output_tokens=metadata.candidates_token_count or 0,
    )


class GeminiClient:
    def __init__(
        self,
        *,
        client: genai.Client | None = None,
        cache: ResponseCache | None = None,
        policy: retry.RetryPolicy | None = None,
        default_model: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or genai.Client(api_key=settings.google_api_key)
        self._cache = cache
        self._policy = policy or retry.RetryPolicy()
        self._default_model = default_model or settings.model_small
        # Injectable so tests exercise the backoff logic without waiting for
        # it. The server's retry hints are tens of seconds; a suite that
        # honours them for real takes minutes and is useless in CI.
        self._sleep = sleep

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        variant: int = 0,
    ) -> Completion:
        """Generate a completion, via the cache when one is configured.

        `variant` distinguishes repeated draws of the same prompt. Without it,
        sampling the same question five times would return five copies of one
        cached response, and any measure of agreement between them would be a
        measurement of the cache.
        """
        chosen = model or self._default_model
        params: dict[str, Any] = {"temperature": temperature, "variant": variant}
        key = cache_key(chosen, prompt, params)

        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                return hit

        completion = self._call(chosen, prompt, temperature)

        if self._cache is not None:
            self._cache.put(key, completion)
        return completion

    def _call(self, model: str, prompt: str, temperature: float) -> Completion:
        attempts = 0
        started = time.perf_counter()

        def once() -> types.GenerateContentResponse:
            nonlocal attempts
            attempts += 1
            try:
                return self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
            except TRANSIENT_NETWORK_ERRORS as error:
                raise retry.TransientError(f"{type(error).__name__}: {error}") from error
            except errors.APIError as error:
                if error.code == 429 and is_daily_quota(error):
                    raise QuotaExhausted(model, str(error)) from error
                if is_transient(error):
                    raise retry.TransientError(
                        str(error), retry_after=retry_after_seconds(error)
                    ) from error
                raise

        response = retry.with_retries(once, self._policy, sleep=self._sleep)
        return Completion(
            text=response.text or "",
            model=model,
            usage=_usage_from(response),
            latency_ms=(time.perf_counter() - started) * 1000,
            cached=False,
            attempts=attempts,
        )


def default_client(cache: ResponseCache | None = None) -> GeminiClient:
    """A client wired to the configured cache."""
    if cache is None:
        cache = ResponseCache(settings.response_cache)
    return GeminiClient(cache=cache)
