"""Download the BIRD Mini-Dev split and report what arrived."""

import argparse
import sys
from pathlib import Path

from askdb.config import settings
from askdb.data import minidev


def report(root: Path) -> int:
    """Describe what is usable in a downloaded dataset."""
    try:
        questions_path = minidev.locate_questions(root)
    except minidev.DatasetError as error:
        print(f"questions: MISSING ({error})", file=sys.stderr)
        return 1

    questions = minidev.load_questions(questions_path)
    databases = minidev.databases_in_use(questions)
    print(f"questions: {len(questions)} in {questions_path}")
    print(f"databases referenced: {len(databases)}")

    try:
        databases_root = minidev.locate_databases(root)
    except minidev.DatasetError as error:
        print(f"databases: MISSING ({error})", file=sys.stderr)
        return 1

    missing = []
    for db_id in databases:
        try:
            minidev.resolve_database(databases_root, db_id)
        except minidev.DatasetError:
            missing.append(db_id)

    found = len(databases) - len(missing)
    print(f"databases present: {found}/{len(databases)} under {databases_root}")
    if missing:
        print(f"missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=settings.data_dir / "mini_dev",
        help="where to place the dataset (default: data/mini_dev)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="only report on what is already present",
    )
    args = parser.parse_args(argv)

    if not args.skip_download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print(
                'huggingface-hub is not installed. Run: pip install -e ".[data]"',
                file=sys.stderr,
            )
            return 1

        args.destination.mkdir(parents=True, exist_ok=True)
        print(f"downloading {minidev.HUGGINGFACE_DATASET} into {args.destination}")
        snapshot_download(
            repo_id=minidev.HUGGINGFACE_DATASET,
            repo_type="dataset",
            local_dir=str(args.destination),
        )

    return report(args.destination)


if __name__ == "__main__":
    raise SystemExit(main())
