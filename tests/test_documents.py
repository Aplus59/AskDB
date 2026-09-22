from pathlib import Path

from askdb.db import catalog
from askdb.retrieval import documents


def test_splits_snake_case() -> None:
    assert documents.tokenize("sch_name") == ["sch", "name"]


def test_splits_camel_case() -> None:
    assert documents.tokenize("FreeMealCount") == ["free", "meal", "count"]


def test_splits_acronym_from_following_word() -> None:
    # BIRD's california_schools database has a column named exactly this.
    assert documents.tokenize("CDSCode") == ["cds", "code"]


def test_keeps_acronyms_together() -> None:
    assert documents.tokenize("URL") == ["url"]


def test_strips_punctuation_and_keeps_digits() -> None:
    assert documents.tokenize("Free Meal Count (K-12)") == ["free", "meal", "count", "k", "12"]


def test_empty_text_yields_no_tokens() -> None:
    assert documents.tokenize("   ") == []


def test_document_weights_the_table_name(sample_db: Path) -> None:
    artist = catalog.load(sample_db).table("artist")
    assert artist is not None

    document = documents.describe(artist, table_name_weight=3)
    # Three from the weighted name, one more from the artist_id column.
    assert document.tokens.count("artist") == 4


def test_document_includes_column_names(sample_db: Path) -> None:
    artist = catalog.load(sample_db).table("artist")
    assert artist is not None

    tokens = documents.describe(artist).tokens
    assert "country" in tokens
    assert "name" in tokens


def test_document_includes_foreign_key_targets(sample_db: Path) -> None:
    # A question naming "album" should be able to reach the track table.
    track = catalog.load(sample_db).table("track")
    assert track is not None
    assert "album" in documents.describe(track).tokens


def test_describes_every_table(sample_db: Path) -> None:
    result = documents.describe_catalog(catalog.load(sample_db))
    assert [document.table for document in result] == ["album", "artist", "track"]
