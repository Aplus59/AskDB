from pathlib import Path

import pytest

from askdb.db import catalog
from askdb.retrieval import documents, lexical


@pytest.fixture
def index(sample_db: Path) -> lexical.BM25:
    return lexical.BM25(documents.describe_catalog(catalog.load(sample_db)))


def test_indexes_every_table(index: lexical.BM25) -> None:
    assert len(index) == 3


def test_unique_column_selects_its_table(index: lexical.BM25) -> None:
    # Only artist has a country column, so nothing else should score at all.
    results = index.search("which country is the musician from")
    assert results[0].table == "artist"
    assert results[0].score > 0
    assert all(result.score == 0 for result in results[1:])


def test_table_name_outranks_a_foreign_key_mention(index: lexical.BM25) -> None:
    # album references artist, so it matches too, but weaker than artist itself.
    results = index.search("artist")
    assert [result.table for result in results[:2]] == ["artist", "album"]
    assert results[0].score > results[1].score


def test_limit_truncates_results(index: lexical.BM25) -> None:
    assert len(index.search("artist", limit=2)) == 2


def test_no_limit_returns_every_table(index: lexical.BM25) -> None:
    assert len(index.search("artist")) == 3


def test_unmatched_question_ranks_deterministically(index: lexical.BM25) -> None:
    # With every score tied at zero, ordering falls back to the table name so
    # that repeated evaluation runs stay comparable.
    results = index.search("xyzzy")
    assert [result.table for result in results] == ["album", "artist", "track"]
    assert all(result.score == 0 for result in results)


def test_empty_index_returns_nothing() -> None:
    assert lexical.BM25([]).search("anything") == []
