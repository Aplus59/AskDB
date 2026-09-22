import sqlite3
from pathlib import Path

import pytest

from askdb.db import catalog
from askdb.db.connection import DatabaseError, read_only


def test_loads_every_table(sample_db: Path) -> None:
    result = catalog.load(sample_db)
    assert [t.name for t in result.tables] == ["album", "artist", "track"]


def test_reads_column_types_and_constraints(sample_db: Path) -> None:
    artist = catalog.load(sample_db).table("artist")
    assert artist is not None

    artist_id = artist.column("artist_id")
    assert artist_id is not None
    assert artist_id.type == "INTEGER"
    assert artist_id.primary_key

    name = artist.column("name")
    assert name is not None
    assert not name.nullable

    country = artist.column("country")
    assert country is not None
    assert country.nullable


def test_table_lookup_is_case_insensitive(sample_db: Path) -> None:
    result = catalog.load(sample_db)
    assert result.table("ARTIST") is not None
    assert result.table("missing") is None


def test_resolves_explicit_foreign_key(sample_db: Path) -> None:
    album = catalog.load(sample_db).table("album")
    assert album is not None
    assert album.foreign_keys == (
        catalog.ForeignKey(
            column="artist_id", references_table="artist", references_column="artist_id"
        ),
    )


def test_resolves_foreign_key_that_omits_target_column(sample_db: Path) -> None:
    # `REFERENCES album` without a column should resolve to album's primary key.
    track = catalog.load(sample_db).table("track")
    assert track is not None
    assert track.foreign_keys == (
        catalog.ForeignKey(
            column="album_id", references_table="album", references_column="album_id"
        ),
    )


def test_counts_rows(sample_db: Path) -> None:
    result = catalog.load(sample_db)
    counts = {t.name: t.row_count for t in result.tables}
    assert counts == {"artist": 3, "album": 2, "track": 3}


def test_ddl_can_be_restricted_to_chosen_tables(sample_db: Path) -> None:
    result = catalog.load(sample_db)
    ddl = result.to_ddl(tables=["artist"])
    assert "CREATE TABLE artist" in ddl
    assert "CREATE TABLE album" not in ddl


def test_ddl_reports_row_counts_and_keys(sample_db: Path) -> None:
    ddl = catalog.load(sample_db).to_ddl(tables=["album"])
    assert "-- 2 rows" in ddl
    assert "FOREIGN KEY (artist_id) REFERENCES artist(artist_id)" in ddl


def test_sample_values_returns_distinct_non_null(sample_db: Path) -> None:
    with read_only(sample_db) as conn:
        values = catalog.sample_values(conn, "artist", "country")
    assert sorted(values) == ["ML", "US"]


def test_sample_values_respects_limit(sample_db: Path) -> None:
    with read_only(sample_db) as conn:
        values = catalog.sample_values(conn, "artist", "name", limit=2)
    assert len(values) == 2


def test_connection_rejects_writes(sample_db: Path) -> None:
    with read_only(sample_db) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM artist")


def test_missing_database_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(DatabaseError, match="database not found"), read_only(
        tmp_path / "nope.sqlite"
    ):
        pass
