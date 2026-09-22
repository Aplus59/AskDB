"""Loading the BIRD Mini-Dev split.

Mini-Dev is the small BIRD split used for day-to-day iteration. The full dev
split is 33.4 GB and is only run for final numbers.
"""

import json
from dataclasses import dataclass
from pathlib import Path

HUGGINGFACE_DATASET = "birdsql/bird_mini_dev"


class DatasetError(Exception):
    """Raised when the dataset is missing or malformed."""


@dataclass(frozen=True)
class Question:
    question_id: int
    db_id: str
    question: str
    gold_sql: str
    evidence: str = ""
    difficulty: str | None = None


def load_questions(path: Path) -> tuple[Question, ...]:
    """Parse `mini_dev_sqlite.json` into typed records."""
    if not path.is_file():
        raise DatasetError(
            f"questions file not found: {path}\n"
            f"Download the dataset from https://huggingface.co/datasets/{HUGGINGFACE_DATASET}"
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise DatasetError(f"expected a list of questions in {path}, got {type(raw).__name__}")

    return tuple(_parse(entry, index) for index, entry in enumerate(raw))


def _parse(entry: dict[str, object], index: int) -> Question:
    missing = [field for field in ("db_id", "question", "SQL") if field not in entry]
    if missing:
        raise DatasetError(f"question at index {index} is missing fields: {', '.join(missing)}")

    raw_id = entry.get("question_id", index)
    return Question(
        question_id=int(raw_id) if isinstance(raw_id, int | str) else index,
        db_id=str(entry["db_id"]),
        question=str(entry["question"]),
        gold_sql=str(entry["SQL"]),
        evidence=str(entry.get("evidence") or ""),
        difficulty=str(entry["difficulty"]) if entry.get("difficulty") else None,
    )


def resolve_database(root: Path, db_id: str) -> Path:
    """Find the SQLite file for a database id.

    The published archives have not been consistent about whether the file sits
    directly in the database folder or under a `sqlite/` subfolder, so try the
    documented locations and fall back to searching.
    """
    folder = root / db_id
    candidates = [folder / f"{db_id}.sqlite", folder / "sqlite" / f"{db_id}.sqlite"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if folder.is_dir():
        found = sorted(folder.rglob("*.sqlite"))
        if found:
            return found[0]

    raise DatasetError(
        f"no SQLite file for database {db_id!r} under {folder}. "
        f"Expected one of: {', '.join(str(c) for c in candidates)}"
    )


def databases_in_use(questions: tuple[Question, ...]) -> tuple[str, ...]:
    return tuple(sorted({q.db_id for q in questions}))
