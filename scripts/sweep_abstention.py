"""Sweep the abstention thresholds over a finished evaluation run.

Costs nothing and calls nothing: the confidence score is recomputed from
signals already stored per question, so every operating point comes out of the
same run.
"""

import argparse
import sys
from pathlib import Path

from askdb.config import settings
from askdb.eval import abstention, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=settings.data_dir / "runs" / "agent.jsonl"
    )
    parser.add_argument(
        "--refuse-below",
        type=float,
        default=0.3,
        help="confidence under which the system declines outright",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    args = parser.parse_args(argv)

    results = list(run.load_results(args.results).values())
    if not results:
        print(f"no results in {args.results}", file=sys.stderr)
        return 1

    thresholds = args.thresholds or [0.0, 0.3, 0.45, 0.6, 0.75, 0.85, 0.95, 1.01]
    points = abstention.sweep(results, thresholds, refuse_below=args.refuse_below)

    print(f"abstention sweep over {len(results)} questions from {args.results}")
    print()
    print(abstention.render(points))
    print()
    print("clarify< 0.00 answers everything; 1.01 answers nothing.")
    print("silent err = answered confidently and wrong, as a share of all questions.")
    print("abst prec  = of the questions declined, the share that would have been wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
