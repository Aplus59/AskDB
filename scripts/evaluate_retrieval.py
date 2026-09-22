"""Measure BM25 schema retrieval on BIRD Mini-Dev.

This produces the baseline every later retrieval change is compared against.
Run it before adding embeddings, and again afterwards.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from askdb.config import settings
from askdb.data import minidev
from askdb.data.minidev import Question
from askdb.db import catalog
from askdb.eval import retrieval
from askdb.retrieval import documents, lexical


def group_by_database(questions: tuple[Question, ...]) -> dict[str, list[Question]]:
    grouped: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        grouped[question.db_id].append(question)
    return dict(grouped)


def query_text(question: Question, use_evidence: bool) -> str:
    """What gets handed to the retriever.

    BIRD's `evidence` field is expert-written context that often names the
    exact columns involved, so including it should help. Whether it actually
    does is worth measuring rather than assuming, hence the flag.
    """
    if use_evidence and question.evidence:
        return f"{question.question} {question.evidence}"
    return question.question


def collect_outcomes(
    questions: tuple[Question, ...], databases_root: Path, use_evidence: bool
) -> tuple[list[retrieval.Outcome], dict[str, list[retrieval.Outcome]]]:
    all_outcomes: list[retrieval.Outcome] = []
    per_database: dict[str, list[retrieval.Outcome]] = {}

    for db_id, group in sorted(group_by_database(questions).items()):
        try:
            database_path = minidev.resolve_database(databases_root, db_id)
            schema = catalog.load(database_path)
        except Exception as error:  # noqa: BLE001 - one bad database must not stop the run
            print(f"skipping {db_id}: {error}", file=sys.stderr)
            continue

        index = lexical.BM25(documents.describe_catalog(schema))
        known_tables = [table.name for table in schema.tables]

        outcomes = [
            retrieval.Outcome(
                question_id=question.question_id,
                gold_tables=retrieval.tables_mentioned(question.gold_sql, known_tables),
                ranked=tuple(
                    result.table for result in index.search(query_text(question, use_evidence))
                ),
            )
            for question in group
        ]
        per_database[db_id] = outcomes
        all_outcomes.extend(outcomes)

    return all_outcomes, per_database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.data_dir / "mini_dev")
    parser.add_argument("--limit", type=int, default=None, help="only score the first N questions")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--with-evidence",
        action="store_true",
        help="append BIRD's evidence field to the retrieval query",
    )
    args = parser.parse_args(argv)

    try:
        questions_path = minidev.locate_questions(args.data)
        databases_root = minidev.locate_databases(args.data)
    except minidev.DatasetError as error:
        print(error, file=sys.stderr)
        print("Run scripts/fetch_minidev.py first.", file=sys.stderr)
        return 1

    questions = minidev.load_questions(questions_path)
    if args.limit is not None:
        questions = questions[: args.limit]

    outcomes, per_database = collect_outcomes(questions, databases_root, args.with_evidence)
    if not outcomes:
        print("no questions could be scored", file=sys.stderr)
        return 1

    report = retrieval.evaluate(outcomes, k_values=args.k)

    print()
    print(f"BM25 schema retrieval{' + evidence' if args.with_evidence else ''}")
    print(report.summary())
    print()

    print("per database:")
    width = max(len(db_id) for db_id in per_database)
    smallest_k = min(args.k)
    largest_k = max(args.k)
    for db_id, group in sorted(per_database.items()):
        breakdown = retrieval.evaluate(group, k_values=args.k)
        print(
            f"  {db_id:<{width}}  "
            f"@{smallest_k}={breakdown.recall[smallest_k]:.3f}  "
            f"@{largest_k}={breakdown.recall[largest_k]:.3f}  "
            f"(n={breakdown.scored})"
        )

    depths = [outcome.depth_needed() for outcome in outcomes]
    unreachable = sum(1 for depth in depths if depth is None)
    if unreachable:
        print()
        print(f"never retrievable at any depth: {unreachable}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
