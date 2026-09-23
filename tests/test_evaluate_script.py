"""The stop/continue decision in the evaluation script.

This branch had no test, and an edit to it silently failed to apply. Every
linter and the whole suite stayed green, and a 500-question run then halted at
453 on a transient 503 it should have ignored, losing an entire database.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from askdb.llm.client import QuotaExhausted
from askdb.llm.fallback import NoModelAvailable
from askdb.llm.retry import RetriesExhausted, TransientError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_agent.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_agent", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_agent"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return load_script()


def saturated() -> RetriesExhausted:
    return RetriesExhausted(5, TransientError("503 UNAVAILABLE"))


def test_quota_everywhere_stops_the_run(script: ModuleType) -> None:
    error = NoModelAvailable(
        ["a", "b"], [QuotaExhausted("a", "daily"), QuotaExhausted("b", "daily")]
    )
    assert script.should_stop(error)


def test_transient_saturation_does_not_stop_the_run(script: ModuleType) -> None:
    # The exact case that cost 47 questions.
    error = NoModelAvailable(["a", "b"], [saturated(), saturated()])
    assert not script.should_stop(error)


def test_a_mix_does_not_stop_the_run(script: ModuleType) -> None:
    error = NoModelAvailable(["a", "b"], [saturated(), QuotaExhausted("b", "daily")])
    assert not script.should_stop(error)
