from pathlib import Path

import pytest

from askdb.llm import cache
from askdb.llm.types import Completion, Usage


@pytest.fixture
def store(tmp_path: Path) -> cache.ResponseCache:
    return cache.ResponseCache(tmp_path / "responses.sqlite")


def a_completion(text: str = "SELECT 1") -> Completion:
    return Completion(
        text=text,
        model="gemini-flash",
        usage=Usage(input_tokens=120, output_tokens=8),
    )


def test_missing_key_returns_none(store: cache.ResponseCache) -> None:
    assert store.get("absent") is None


def test_round_trips_a_completion(store: cache.ResponseCache) -> None:
    store.put("k", a_completion())
    restored = store.get("k")

    assert restored is not None
    assert restored.text == "SELECT 1"
    assert restored.model == "gemini-flash"
    assert restored.usage.input_tokens == 120
    assert restored.usage.output_tokens == 8


def test_restored_completions_are_marked_cached(store: cache.ResponseCache) -> None:
    # Callers need to tell a cache hit from a real call when attributing cost.
    store.put("k", a_completion())
    restored = store.get("k")
    assert restored is not None
    assert restored.cached


def test_writing_the_same_key_replaces_it(store: cache.ResponseCache) -> None:
    store.put("k", a_completion("first"))
    store.put("k", a_completion("second"))

    restored = store.get("k")
    assert restored is not None
    assert restored.text == "second"
    assert len(store) == 1


def test_clear_empties_the_store(store: cache.ResponseCache) -> None:
    store.put("k", a_completion())
    store.clear()
    assert len(store) == 0


def test_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "responses.sqlite"
    cache.ResponseCache(path).put("k", a_completion())

    reopened = cache.ResponseCache(path)
    assert reopened.get("k") is not None


def test_creates_its_parent_directory(tmp_path: Path) -> None:
    store = cache.ResponseCache(tmp_path / "nested" / "deeper" / "responses.sqlite")
    store.put("k", a_completion())
    assert store.get("k") is not None


def test_key_is_stable_across_calls() -> None:
    assert cache.cache_key("m", "prompt") == cache.cache_key("m", "prompt")


def test_key_ignores_parameter_ordering() -> None:
    # Two identical calls should share an entry regardless of dict ordering.
    left = cache.cache_key("m", "prompt", {"temperature": 0, "top_p": 1})
    right = cache.cache_key("m", "prompt", {"top_p": 1, "temperature": 0})
    assert left == right


def test_key_changes_with_the_model() -> None:
    assert cache.cache_key("flash", "prompt") != cache.cache_key("pro", "prompt")


def test_key_changes_with_the_prompt() -> None:
    assert cache.cache_key("m", "one") != cache.cache_key("m", "two")


def test_key_changes_with_parameters() -> None:
    assert cache.cache_key("m", "p", {"temperature": 0}) != cache.cache_key(
        "m", "p", {"temperature": 1}
    )


def test_usage_totals_and_adds() -> None:
    left = Usage(input_tokens=10, output_tokens=2)
    right = Usage(input_tokens=5, output_tokens=3)

    assert left.total_tokens == 12
    assert (left + right) == Usage(input_tokens=15, output_tokens=5)
