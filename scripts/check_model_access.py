"""Confirm the API key works and both rungs of the cost ladder are reachable.

Makes one small real call per model. Run this before anything expensive, so a
misconfigured key fails in two seconds rather than halfway through an
evaluation.
"""

import argparse
import sys

from askdb.config import settings
from askdb.llm.client import GeminiClient

PROMPT = "Reply with exactly the word: ready"


def check(client: GeminiClient, model: str) -> bool:
    try:
        completion = client.complete(PROMPT, model=model)
    except Exception as error:  # noqa: BLE001 - report any failure, do not raise
        print(f"  {model:<28} FAILED  {type(error).__name__}: {error}")
        return False

    print(
        f"  {model:<28} ok      "
        f"{completion.usage.total_tokens:>4} tokens  "
        f"{completion.latency_ms:>7.0f} ms  "
        f"{completion.attempts} attempt(s)  "
        f"{'(cached)' if completion.cached else ''}"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="check a specific model instead")
    args = parser.parse_args(argv)

    if not settings.google_api_key:
        print("GOOGLE_API_KEY is not set. Put it in .env at the repository root.", file=sys.stderr)
        return 1

    models = args.model or [settings.model_small, settings.model_large]

    # No cache here: the point is to prove the network path works.
    client = GeminiClient()
    print("checking model access")
    results = [check(client, model) for model in models]

    if not all(results):
        print("\nat least one model is unreachable", file=sys.stderr)
        return 1

    print("\nall models reachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
