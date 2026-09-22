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


QUESTIONS_FILENAME = "mini_dev_sqlite.json"
# HuggingFace serves the same file shard-named, e.g. mini_dev_sqlite-00000-of-00001.json
QUESTIONS_GLOB = "mini_dev_sqlite*.json"
DATABASES_DIRNAME = "dev_databases"

# The databases are not published on HuggingFace, only the questions are.
DATABASES_URL = "https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view"


def locate_questions(root: Path) -> Path:
    """Find the questions file inside a downloaded dataset.

    The archive has been repackaged more than once and HuggingFace adds shard
    suffixes to the name, so search rather than assume a fixed layout.
    """
    direct = root / QUESTIONS_FILENAME
    if direct.is_file():
        return direct

    found = sorted(root.rglob(QUESTIONS_GLOB))
    if not found:
        raise DatasetError(f"no {QUESTIONS_GLOB} found under {root}")
    return found[0]


def locate_databases(root: Path) -> Path:
    """Find the folder holding the per-database SQLite files."""
    direct = root / DATABASES_DIRNAME
    if direct.is_dir():
        return direct

    found = sorted(path for path in root.rglob(DATABASES_DIRNAME) if path.is_dir())
    if not found:
        raise DatasetError(
            f"no {DATABASES_DIRNAME} folder under {root}.\n"
            "The questions and the databases are distributed separately: HuggingFace "
            f"carries only the questions. Download the databases from {DATABASES_URL} "
            f"and extract them so that a {DATABASES_DIRNAME} folder sits under {root}."
        )
    return found[0]
