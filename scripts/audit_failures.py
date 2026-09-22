"""Inspect and classify the failures from an evaluation run.

Three modes:

    --list                    show each unlabelled failure with its evidence
    --label ID CATEGORY       record a judgement
    --summary                 show the taxonomy and corrected accuracy

The listing runs both queries and prints the rows each one returned. That is
the evidence the judgement rests on: two queries can look completely different
and return identical data, or look nearly identical and differ in one column.
Reading the SQL alone is not enough.
"""

import argparse
import sys
from pathlib import Path

from askdb.config import settings
from askdb.data import minidev
from askdb.eval import audit, execution, run

PREVIEW_ROWS = 5


def show_rows(database: Path, sql: str, label: str) -> None:
    result = execution.run(database, sql, max_rows=PREVIEW_ROWS)
    if not result.ok:
        print(f"    {label}: FAILED {result.error}")
        return

    rows = result.rows or ()
    if not rows:
        print(f"    {label}: no rows")
        return

    shown = ", ".join(str(cell) for cell in rows[0])
    more = f"  (+{len(rows) - 1} more)" if len(rows) > 1 else ""
    columns = "/".join(result.columns) if result.columns else "?"
    print(f"    {label}: [{columns}] {shown[:90]}{more}")


def list_failures(
    results: list[run.QuestionResult],
    labels: dict[int, audit.Label],
    databases_root: Path,
    include_labelled: bool,
    limit: int | None,
) -> None:
    failures = [result for result in results if not result.match]
    if not include_labelled:
        failures = [f for f in failures if f.question_id not in labels]
    if limit is not None:
        failures = failures[:limit]

    if not failures:
        print("nothing to review")
        return

    for failure in failures:
        try:
            database = minidev.resolve_database(databases_root, failure.db_id)
        except minidev.DatasetError as error:
            print(f"q{failure.question_id}: cannot open database ({error})")
            continue

        existing = labels.get(failure.question_id)
        print(f"q{failure.question_id}  [{failure.db_id}]  {failure.failure_reason}")
        if existing:
            note = f" — {existing.note}" if existing.note else ""
            print(f"  labelled: {existing.category}{note}")
        print(f"  Q    {failure.question}")
        print(f"  gold {' '.join(failure.gold_sql.split())[:160]}")
        print(f"  pred {' '.join((failure.predicted_sql or '').split())[:160]}")
        show_rows(database, failure.gold_sql, "gold")
        if failure.predicted_sql:
            show_rows(database, failure.predicted_sql, "pred")
        print()

    print(f"{len(failures)} shown. Categories: {', '.join(audit.CATEGORIES)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.data_dir / "mini_dev")
    parser.add_argument("--results", type=Path, default=settings.data_dir / "runs" / "agent.jsonl")
    parser.add_argument("--labels", type=Path, default=settings.data_dir / "runs" / "labels.jsonl")
    parser.add_argument("--list", action="store_true", help="show failures needing review")
    parser.add_argument("--all", action="store_true", help="include already-labelled failures")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--label",
        nargs=2,
        metavar=("QUESTION_ID", "CATEGORY"),
        help="record a judgement for one question",
    )
    parser.add_argument("--note", default="", help="reasoning to store with the label")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)

    results = list(run.load_results(args.results).values())
    if not results:
        print(f"no results in {args.results}", file=sys.stderr)
        return 1

    labels = audit.load_labels(args.labels)

    if args.label:
        raw_id, category = args.label
        try:
            label = audit.Label(question_id=int(raw_id), category=category, note=args.note)
        except (ValueError, audit.UnknownCategory) as error:
            print(error, file=sys.stderr)
            return 1

        audit.save_label(args.labels, label)
        print(f"q{label.question_id} labelled {label.category}")
        labels[label.question_id] = label

    if args.list:
        try:
            databases_root = minidev.locate_databases(args.data)
        except minidev.DatasetError as error:
            print(error, file=sys.stderr)
            return 1
        list_failures(results, labels, databases_root, args.all, args.limit)
        return 0

    print(audit.summarize(results, labels).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
