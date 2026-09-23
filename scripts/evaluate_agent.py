"""Run the agent over BIRD Mini-Dev and report execution accuracy.

Safe to interrupt. Results append to the output file as they are produced, and
re-running with the same file skips questions already answered.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from askdb.agent.loop import Agent
from askdb.agent.tools import Toolbox
from askdb.config import settings
from askdb.data import minidev
from askdb.data.minidev import Question
from askdb.db import catalog
from askdb.eval import run
from askdb.llm.factory import build_client


def group_by_database(questions: list[Question]) -> dict[str, list[Question]]:
    grouped: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        grouped[question.db_id].append(question)
    return dict(grouped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.data_dir / "mini_dev")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir / "runs" / "agent.jsonl",
        help="where per-question results are appended",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="answer at most N questions THIS RUN, spread across databases. "
        "When resuming, this is N more on top of what the output file already "
        "holds, not a total.",
    )
    parser.add_argument("--db", action="append", help="restrict to these databases")
    parser.add_argument("--model", default=None, help="override the model")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="draw the first query N times and keep the most agreed-upon answer; "
        "costs N times the tokens but yields a continuous confidence signal",
    )
    parser.add_argument("--schema-tables", type=int, default=5)
    parser.add_argument(
        "--value-grounding",
        action="store_true",
        help="include example column values in the schema; measured at +77%% tokens "
        "for no accuracy change (see docs/agent-experiments.md)",
    )
    parser.add_argument(
        "--no-self-check",
        action="store_true",
        help="do not re-examine successful queries whose result looks wrong",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="ignore previous results in the output file"
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="summarise the output file without running"
    )
    args = parser.parse_args(argv)

    try:
        questions_path = minidev.locate_questions(args.data)
        databases_root = minidev.locate_databases(args.data)
    except minidev.DatasetError as error:
        print(error, file=sys.stderr)
        return 1

    done = {} if args.fresh else run.load_results(args.output)

    if args.summary_only:
        if not done:
            print(f"no results in {args.output}", file=sys.stderr)
            return 1
        print(run.summarize(list(done.values())).render())
        return 0

    questions = list(minidev.load_questions(questions_path))
    if args.db:
        wanted = set(args.db)
        questions = [question for question in questions if question.db_id in wanted]

    remaining = run.pending(questions, done)
    if args.limit is not None:
        remaining = run.stratified_sample(remaining, args.limit)

    if done:
        print(f"resuming: {len(done)} already answered, {len(remaining)} to go")
    else:
        print(f"{len(remaining)} questions to answer")

    client = build_client()
    answered = 0

    for db_id, group in sorted(group_by_database(remaining).items()):
        try:
            database = minidev.resolve_database(databases_root, db_id)
            toolbox = Toolbox(database, catalog.load(database))
        except Exception as error:  # noqa: BLE001 - one bad database must not stop the run
            print(f"skipping {db_id}: {error}", file=sys.stderr)
            continue

        agent = Agent(
            toolbox,
            client,
            model=args.model,
            max_repairs=args.max_repairs,
            schema_tables=args.schema_tables,
            ground_values=args.value_grounding,
            self_check=not args.no_self_check,
            samples=args.samples,
        )

        for question in group:
            try:
                result = run.run_question(agent, question, database)
            except KeyboardInterrupt:
                print("\ninterrupted; progress is saved", file=sys.stderr)
                raise
            except Exception as error:  # noqa: BLE001 - keep going, record nothing
                print(f"  q{question.question_id} errored: {error}", file=sys.stderr)
                continue

            run.append_result(args.output, result)
            done[result.question_id] = result
            answered += 1

            mark = "ok " if result.match else "   "
            repairs = f" +{result.repairs}" if result.repairs else "   "
            print(
                f"  {mark}q{result.question_id:<5} {db_id:<24}{repairs} "
                f"{result.total_tokens:>6} tok",
                flush=True,
            )

    print()
    print(f"answered {answered} this run; {len(done)} total in {args.output}")
    print()
    print(run.summarize(list(done.values())).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
