from pathlib import Path

import pytest

from askdb.agent.tools import Toolbox, score_column
from askdb.db import catalog
from askdb.retrieval.documents import tokenize


@pytest.fixture
def toolbox(sample_db: Path) -> Toolbox:
    return Toolbox(sample_db, catalog.load(sample_db))


def test_lists_every_table_with_shape(toolbox: Toolbox) -> None:
    listing = toolbox.list_tables()
    names = {summary.name: summary for summary in listing.tables}

    assert set(names) == {"artist", "album", "track"}
    assert names["artist"].row_count == 3
    assert names["artist"].column_count == 3


def test_table_listing_renders_counts(toolbox: Toolbox) -> None:
    rendered = toolbox.list_tables().render()
    assert "artist (3 columns, 3 rows)" in rendered


def test_describes_columns_with_samples(toolbox: Toolbox) -> None:
    description = toolbox.describe_table("artist")
    assert description.table is not None
    assert description.samples["country"]

    rendered = description.render()
    assert "TABLE artist (3 rows)" in rendered
    assert "country TEXT" in rendered
    assert "e.g." in rendered


def test_describes_primary_keys_and_foreign_keys(toolbox: Toolbox) -> None:
    rendered = toolbox.describe_table("album").render()
    assert "primary key" in rendered
    assert "FOREIGN KEY artist_id -> artist.artist_id" in rendered


def test_unknown_table_lists_what_exists_instead_of_failing(toolbox: Toolbox) -> None:
    # The agent has to be able to recover, so this is a message, not an error.
    description = toolbox.describe_table("playlist")
    assert description.table is None

    rendered = description.render()
    assert "No such table" in rendered
    assert "artist" in rendered


def test_describe_is_case_insensitive(toolbox: Toolbox) -> None:
    assert toolbox.describe_table("ARTIST").table is not None


def test_searches_columns_across_tables(toolbox: Toolbox) -> None:
    matches = toolbox.search_columns("country")
    assert matches.matches[0].table == "artist"
    assert matches.matches[0].column == "country"


def test_search_finds_columns_named_differently_from_the_term(toolbox: Toolbox) -> None:
    matches = toolbox.search_columns("artist id")
    found = {(m.table, m.column) for m in matches.matches}
    assert ("artist", "artist_id") in found
    assert ("album", "artist_id") in found


def test_search_returns_nothing_for_an_absent_concept(toolbox: Toolbox) -> None:
    assert toolbox.search_columns("zzzz").matches == ()


def test_search_ranking_is_deterministic(toolbox: Toolbox) -> None:
    first = toolbox.search_columns("title").matches
    second = toolbox.search_columns("title").matches
    assert first == second


def test_search_respects_its_limit(toolbox: Toolbox) -> None:
    assert len(toolbox.search_columns("id", limit=2).matches) <= 2


def test_exact_column_name_scores_highest() -> None:
    assert score_column(tokenize("country"), "country") == 1.0


def test_token_overlap_scores_above_zero() -> None:
    score = score_column(tokenize("artist name"), "artist_id")
    assert 0 < score < 1.0


def test_unrelated_column_scores_zero() -> None:
    assert score_column(tokenize("weather"), "artist_id") == 0.0


def test_samples_values_from_a_column(toolbox: Toolbox) -> None:
    assert sorted(toolbox.sample_values("artist", "country")) == ["ML", "US"]


def test_sampling_an_unknown_column_returns_nothing(toolbox: Toolbox) -> None:
    assert toolbox.sample_values("artist", "nonexistent") == ()
    assert toolbox.sample_values("nonexistent", "country") == ()


def test_executes_a_query_and_names_the_columns(toolbox: Toolbox) -> None:
    result = toolbox.execute_sql("SELECT name, country FROM artist ORDER BY name")

    assert result.ok
    assert result.columns == ("name", "country")
    assert result.rows[0] == ("Ali Farka Toure", "ML")


def test_query_renders_as_a_table(toolbox: Toolbox) -> None:
    rendered = toolbox.execute_sql("SELECT name FROM artist ORDER BY name").render()
    assert rendered.splitlines()[0] == "name"
    assert "Miles Davis" in rendered


def test_invalid_sql_comes_back_as_a_readable_message(toolbox: Toolbox) -> None:
    # The repair loop needs the database's own words, not a stack trace.
    result = toolbox.execute_sql("SELECT nope FROM artist")

    assert not result.ok
    assert result.error is not None
    assert "nope" in result.error
    assert "Query failed" in result.render()


def test_empty_results_say_so(toolbox: Toolbox) -> None:
    result = toolbox.execute_sql("SELECT name FROM artist WHERE country = 'XX'")
    assert result.ok
    assert result.render() == "Query returned no rows."


def test_results_are_capped_and_the_cap_is_visible(sample_db: Path) -> None:
    box = Toolbox(sample_db, catalog.load(sample_db), max_rows=2)
    result = box.execute_sql("SELECT name FROM artist")

    assert len(result.rows) == 2
    assert result.truncated
    assert "more available" in result.render()


def test_a_write_is_rejected_by_the_connection(toolbox: Toolbox) -> None:
    # Read-only is enforced by the driver, so this is reported, not executed.
    result = toolbox.execute_sql("DELETE FROM artist")
    assert not result.ok


def test_schema_context_shows_example_values(toolbox: Toolbox) -> None:
    # The measured failure this exists for: the model filtered on
    # 'East Bohemia' while the column stored 'east Bohemia'.
    context = toolbox.schema_context(["artist"])

    assert "CREATE TABLE artist" in context
    assert "e.g." in context
    assert "'US'" in context or "'ML'" in context


def test_schema_context_omits_tables_not_asked_for(toolbox: Toolbox) -> None:
    context = toolbox.schema_context(["artist"])
    assert "CREATE TABLE track" not in context


def test_schema_context_keeps_foreign_keys(toolbox: Toolbox) -> None:
    context = toolbox.schema_context(["album"])
    assert "FOREIGN KEY (artist_id) REFERENCES artist(artist_id)" in context


def test_schema_context_reports_row_counts(toolbox: Toolbox) -> None:
    assert "-- 3 rows" in toolbox.schema_context(["artist"])


def test_numeric_columns_get_no_examples(toolbox: Toolbox) -> None:
    # A filter on a number is written from the question; showing values only
    # spends tokens.
    context = toolbox.schema_context(["album"])
    released = next(line for line in context.splitlines() if "released" in line)
    assert "e.g." not in released


def test_examples_can_be_switched_off(toolbox: Toolbox) -> None:
    assert "e.g." not in toolbox.schema_context(["artist"], samples_per_column=0)


def test_schema_context_is_cached(toolbox: Toolbox) -> None:
    # Re-sampling per question would mean thousands of scans over tables that
    # reach hundreds of megabytes.
    first = toolbox.schema_context(["artist"])
    toolbox.database = Path("deliberately-invalid")
    assert toolbox.schema_context(["artist"]) == first


def test_text_column_detection() -> None:
    from askdb.agent.tools import is_text_column

    assert is_text_column("TEXT")
    assert is_text_column("VARCHAR(20)")
    assert is_text_column("")
    assert not is_text_column("INTEGER")
    assert not is_text_column("REAL")


def test_long_cells_are_shortened(sample_db: Path) -> None:
    box = Toolbox(sample_db, catalog.load(sample_db))
    result = box.execute_sql("SELECT '" + "x" * 200 + "' AS wide")
    assert len(result.render().splitlines()[1]) < 100
