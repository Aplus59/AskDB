"""Shared model-call types.

Token counts travel with every response rather than being logged separately,
because cost attribution per pipeline step is one of the project's results.
A number that has to be reconstructed afterwards usually is not.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    cached: bool = False
    attempts: int = 1

    # True when the preferred model was unavailable and a fallback answered.
    # Callers can still use the result, but it should not be counted as a
    # measurement of the model that was asked for.
    degraded: bool = False
