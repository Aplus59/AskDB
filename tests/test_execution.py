from pathlib import Path

from askdb.eval import execution


def test_identical_queries_match(sample_db: Path) -> None:
    sql = "SELECT name FROM artist ORDER BY artist_id"
    result = execution.score(sample_db, sql, sql)
    assert result.match
    assert result.exact_match
    assert result.failure_reason is None


def test_row_order_is_ignored(sample_db: Path) -> None:
    result = execution.score(
        sample_db,
        "SELECT name FROM artist ORDER BY name DESC",
        "SELECT name FROM artist ORDER BY name ASC",
    )
    assert result.match


def test_duplicates_collapse_in_headline_metric_but_not_in_exact(sample_db: Path) -> None:
    # `country` holds US twice; SELECT DISTINCT returns it once. The official
    # set-based metric calls these equal, the multiset diagnostic does not.
    result = execution.score(
        sample_db,
        "SELECT country FROM artist",
        "SELECT DISTINCT country FROM artist",
    )
    assert result.match
    assert not result.exact_match


def test_column_order_matters(sample_db: Path) -> None:
    result = execution.score(
        sample_db,
        "SELECT country, name FROM artist",
        "SELECT name, country FROM artist",
    )
    assert not result.match
    assert result.failure_reason == "wrong_rows"


def test_wrong_rows_are_reported(sample_db: Path) -> None:
    result = execution.score(
        sample_db,
        "SELECT name FROM artist WHERE country = 'ML'",
        "SELECT name FROM artist",
    )
    assert not result.match
    assert result.failure_reason == "wrong_rows"


def test_float_differences_below_precision_are_tolerated() -> None:
    match, exact = execution.compare_rows(((1.0000000001,),), ((1.0,),))
    assert match
    assert exact


def test_float_differences_above_precision_are_not() -> None:
    match, _ = execution.compare_rows(((1.01,),), ((1.0,),))
    assert not match


def test_invalid_sql_is_captured_not_raised(sample_db: Path) -> None:
    result = execution.score(
        sample_db,
        "SELECT nope FROM artist",
        "SELECT name FROM artist",
    )
    assert not result.match
    assert result.failure_reason == "execution_error"
    assert result.predicted.error is not None
    assert "nope" in result.predicted.error


def test_runaway_query_is_interrupted(sample_db: Path) -> None:
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT * FROM c"
    result = execution.run(sample_db, runaway, timeout_seconds=0.2)
    assert result.timed_out
    assert result.error == "query timed out"


def test_timeout_surfaces_as_failure_reason(sample_db: Path) -> None:
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT * FROM c"
    result = execution.score(
        sample_db, runaway, "SELECT name FROM artist", timeout_seconds=0.2
    )
    assert not result.match
    assert result.failure_reason == "timeout"


def test_duration_is_recorded(sample_db: Path) -> None:
    result = execution.run(sample_db, "SELECT name FROM artist")
    assert result.ok
    assert result.duration_ms >= 0
