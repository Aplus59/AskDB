import json
from pathlib import Path

import pytest

from askdb.data import minidev

ENTRY = {
    "question_id": 7,
    "db_id": "california_schools",
    "question": "How many schools are in Alameda County?",
    "evidence": "County refers to the County column",
    "SQL": "SELECT COUNT(*) FROM schools WHERE County = 'Alameda'",
    "difficulty": "simple",
}


def write_questions(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    path = tmp_path / "mini_dev_sqlite.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_parses_a_question(tmp_path: Path) -> None:
    questions = minidev.load_questions(write_questions(tmp_path, [ENTRY]))
    assert len(questions) == 1

    question = questions[0]
    assert question.question_id == 7
    assert question.db_id == "california_schools"
    assert question.gold_sql.startswith("SELECT COUNT(*)")
    assert question.evidence == "County refers to the County column"
    assert question.difficulty == "simple"


def test_question_id_falls_back_to_position(tmp_path: Path) -> None:
    entry = {k: v for k, v in ENTRY.items() if k != "question_id"}
    questions = minidev.load_questions(write_questions(tmp_path, [entry, entry]))
    assert [q.question_id for q in questions] == [0, 1]


def test_missing_evidence_becomes_empty_string(tmp_path: Path) -> None:
    entry = {k: v for k, v in ENTRY.items() if k != "evidence"}
    questions = minidev.load_questions(write_questions(tmp_path, [entry]))
    assert questions[0].evidence == ""


def test_missing_required_field_names_the_field(tmp_path: Path) -> None:
    entry = {k: v for k, v in ENTRY.items() if k != "SQL"}
    with pytest.raises(minidev.DatasetError, match="missing fields: SQL"):
        minidev.load_questions(write_questions(tmp_path, [entry]))


def test_absent_file_points_at_the_download(tmp_path: Path) -> None:
    with pytest.raises(minidev.DatasetError, match="huggingface.co"):
        minidev.load_questions(tmp_path / "absent.json")


def test_databases_in_use_is_sorted_and_deduplicated(tmp_path: Path) -> None:
    entries = [
        {**ENTRY, "db_id": "toxicology"},
        {**ENTRY, "db_id": "california_schools"},
        {**ENTRY, "db_id": "toxicology"},
    ]
    questions = minidev.load_questions(write_questions(tmp_path, entries))
    assert minidev.databases_in_use(questions) == ("california_schools", "toxicology")


def test_resolves_database_in_the_flat_layout(tmp_path: Path) -> None:
    folder = tmp_path / "toxicology"
    folder.mkdir()
    expected = folder / "toxicology.sqlite"
    expected.touch()
    assert minidev.resolve_database(tmp_path, "toxicology") == expected


def test_resolves_database_in_the_nested_layout(tmp_path: Path) -> None:
    folder = tmp_path / "toxicology" / "sqlite"
    folder.mkdir(parents=True)
    expected = folder / "toxicology.sqlite"
    expected.touch()
    assert minidev.resolve_database(tmp_path, "toxicology") == expected


def test_falls_back_to_searching_for_any_sqlite_file(tmp_path: Path) -> None:
    folder = tmp_path / "toxicology" / "unexpected"
    folder.mkdir(parents=True)
    expected = folder / "renamed.sqlite"
    expected.touch()
    assert minidev.resolve_database(tmp_path, "toxicology") == expected


def test_missing_database_lists_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(minidev.DatasetError, match="no SQLite file"):
        minidev.resolve_database(tmp_path, "toxicology")


def test_locates_questions_at_the_root(tmp_path: Path) -> None:
    expected = tmp_path / minidev.QUESTIONS_FILENAME
    expected.touch()
    assert minidev.locate_questions(tmp_path) == expected


def test_locates_questions_nested_in_the_archive(tmp_path: Path) -> None:
    nested = tmp_path / "mini_dev_data" / "sqlite"
    nested.mkdir(parents=True)
    expected = nested / minidev.QUESTIONS_FILENAME
    expected.touch()
    assert minidev.locate_questions(tmp_path) == expected


def test_locates_shard_named_questions_file(tmp_path: Path) -> None:
    # HuggingFace serves the file with a shard suffix rather than the plain name.
    nested = tmp_path / "data"
    nested.mkdir()
    expected = nested / "mini_dev_sqlite-00000-of-00001.json"
    expected.touch()
    assert minidev.locate_questions(tmp_path) == expected


def test_plain_name_wins_over_shard_name(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "mini_dev_sqlite-00000-of-00001.json").touch()
    expected = tmp_path / minidev.QUESTIONS_FILENAME
    expected.touch()
    assert minidev.locate_questions(tmp_path) == expected


def test_missing_questions_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(minidev.DatasetError, match=r"no mini_dev_sqlite\*\.json"):
        minidev.locate_questions(tmp_path)


def test_locates_databases_folder(tmp_path: Path) -> None:
    expected = tmp_path / minidev.DATABASES_DIRNAME
    expected.mkdir()
    assert minidev.locate_databases(tmp_path) == expected


def test_locates_nested_databases_folder(tmp_path: Path) -> None:
    expected = tmp_path / "mini_dev_data" / minidev.DATABASES_DIRNAME
    expected.mkdir(parents=True)
    assert minidev.locate_databases(tmp_path) == expected


def test_missing_databases_folder_explains_the_split_distribution(tmp_path: Path) -> None:
    with pytest.raises(minidev.DatasetError, match="distributed separately"):
        minidev.locate_databases(tmp_path)


def test_macos_archive_metadata_is_not_mistaken_for_databases(tmp_path: Path) -> None:
    # Archives zipped on macOS carry a parallel __MACOSX tree. It sorts before
    # the real folder, so without filtering it wins.
    junk = tmp_path / "__MACOSX" / "bundle" / minidev.DATABASES_DIRNAME
    junk.mkdir(parents=True)
    real = tmp_path / "bundle" / minidev.DATABASES_DIRNAME
    real.mkdir(parents=True)

    assert minidev.locate_databases(tmp_path) == real


def test_appledouble_stub_is_not_mistaken_for_a_database(tmp_path: Path) -> None:
    folder = tmp_path / "toxicology"
    folder.mkdir()
    (folder / "._toxicology.sqlite").touch()
    real = folder / "sqlite" / "toxicology.sqlite"
    real.parent.mkdir()
    real.touch()

    assert minidev.resolve_database(tmp_path, "toxicology") == real


def test_appledouble_stub_alone_is_not_accepted(tmp_path: Path) -> None:
    folder = tmp_path / "toxicology"
    folder.mkdir()
    (folder / "._toxicology.sqlite").touch()

    with pytest.raises(minidev.DatasetError, match="no SQLite file"):
        minidev.resolve_database(tmp_path, "toxicology")


def test_questions_file_inside_macos_metadata_is_ignored(tmp_path: Path) -> None:
    junk = tmp_path / "__MACOSX" / "data"
    junk.mkdir(parents=True)
    (junk / "mini_dev_sqlite-00000-of-00001.json").touch()
    real = tmp_path / "data" / "mini_dev_sqlite-00000-of-00001.json"
    real.parent.mkdir()
    real.touch()

    assert minidev.locate_questions(tmp_path) == real
