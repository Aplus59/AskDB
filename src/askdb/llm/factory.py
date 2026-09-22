"""Assembling the model client the rest of the project uses.

Layering, innermost first:

    GeminiClient    cache lookup, then the API call under a retry policy
    FallbackClient  moves to the next model when one is unavailable

Kept separate from both so neither has to import the other.
"""

from collections.abc import Sequence

from askdb.config import settings
from askdb.llm.cache import ResponseCache
from askdb.llm.client import GeminiClient
from askdb.llm.fallback import FallbackClient


def build_client(
    *, cache: ResponseCache | None = None, chain: Sequence[str] | None = None
) -> FallbackClient:
    if cache is None:
        cache = ResponseCache(settings.response_cache)
    inner = GeminiClient(cache=cache)
    return FallbackClient(inner, chain or (settings.model_small, settings.model_large))
