"""Calling Gemini, with caching and retries wrapped around it.

Every call goes through the same path: look in the cache, otherwise call the
API under a retry policy, then record the result. That ordering matters —
a cached response costs nothing and cannot be rate limited, so the cache is
checked before anything else.
"""

import time
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
    ) -> None:
        self._client = client or genai.Client(api_key=settings.google_api_key)
        self._cache = cache
        self._policy = policy or retry.RetryPolicy()
        self._default_model = default_model or settings.model_small

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
                if is_transient(error):
                    raise retry.TransientError(str(error)) from error
                raise

        response = retry.with_retries(once, self._policy)
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
