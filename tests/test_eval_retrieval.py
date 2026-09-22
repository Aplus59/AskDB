import pytest

from askdb.eval import retrieval

KNOWN = ("artist", "album", "track")


def mentioned(sql: str) -> set[str]:
    return set(retrieval.tables_mentioned(sql, KNOWN))


def test_simple_select() -> None:
    assert mentioned("SELECT name FROM artist") == {"artist"}


def test_join_with_aliases() -> None:
    sql = (
        "SELECT T1.name FROM artist AS T1 "
        "JOIN album AS T2 ON T1.artist_id = T2.artist_id"
    )
    assert mentioned(sql) == {"artist", "album"}


def test_comma_separated_tables() -> None:
    assert mentioned("SELECT * FROM artist, album") == {"artist", "album"}


def test_backtick_quoted_name() -> None:
    assert mentioned("SELECT * FROM `album`") == {"album"}


def test_returns_the_catalog_spelling() -> None:
    assert mentioned("select * from ARTIST") == {"artist"}


def test_column_sharing_a_table_name_is_not_counted() -> None:
    # Only FROM and JOIN clauses are read, so a selected column called
    # "album" does not make this query touch the album table.
    assert mentioned("SELECT album FROM track") == {"track"}


def test_string_literal_is_not_a_table() -> None:
    assert mentioned("SELECT * FROM track WHERE title = 'album'") == {"track"}


def test_order_by_does_not_leak_into_table_names() -> None:
    assert mentioned("SELECT * FROM artist ORDER BY name") == {"artist"}


def test_cte_alias_is_ignored_but_its_source_is_not() -> None:
    sql = "WITH recent AS (SELECT * FROM album) SELECT * FROM recent"
    assert mentioned(sql) == {"album"}


def test_subquery_source_is_found() -> None:
    assert mentioned("SELECT * FROM (SELECT * FROM track) AS t") == {"track"}


def test_unknown_table_yields_nothing() -> None:
    assert mentioned("SELECT * FROM playlist") == set()


def test_hit_requires_every_gold_table() -> None:
    outcome = retrieval.Outcome(
        question_id=1,
        gold_tables=frozenset({"artist", "album"}),
        ranked=("artist", "track", "album"),
    )
    assert not outcome.hit_at(2)
    assert outcome.hit_at(3)


def test_hit_is_false_when_gold_is_unknown() -> None:
    outcome = retrieval.Outcome(question_id=1, gold_tables=frozenset(), ranked=("artist",))
    assert not outcome.hit_at(10)


def test_depth_needed_is_the_last_gold_position() -> None:
    outcome = retrieval.Outcome(
        question_id=1,
        gold_tables=frozenset({"artist", "album"}),
        ranked=("artist", "track", "album"),
    )
    assert outcome.depth_needed() == 3


def test_depth_needed_is_none_when_a_table_never_appears() -> None:
    outcome = retrieval.Outcome(
        question_id=1,
        gold_tables=frozenset({"artist", "playlist"}),
        ranked=("artist", "album", "track"),
    )
    assert outcome.depth_needed() is None


def test_recall_across_depths() -> None:
    ranked = ("artist", "album", "track")
    outcomes = [
        retrieval.Outcome(1, frozenset({"artist"}), ranked),
        retrieval.Outcome(2, frozenset({"artist", "album"}), ranked),
        retrieval.Outcome(3, frozenset({"track"}), ranked),
    ]
    report = retrieval.evaluate(outcomes, k_values=(1, 2, 3))

    assert report.recall[1] == pytest.approx(1 / 3)
    assert report.recall[2] == pytest.approx(2 / 3)
    assert report.recall[3] == 1.0
    assert report.scored == 3
    assert report.skipped == 0


def test_unresolved_questions_are_skipped_not_failed() -> None:
    ranked = ("artist",)
    outcomes = [
        retrieval.Outcome(1, frozenset({"artist"}), ranked),
        retrieval.Outcome(2, frozenset(), ranked),
    ]
    report = retrieval.evaluate(outcomes, k_values=(1,))

    # One of two questions could not be scored, but the one that could was
    # correct, so recall is 1.0 over a population of 1 rather than 0.5.
    assert report.recall[1] == 1.0
    assert report.scored == 1
    assert report.skipped == 1


def test_empty_input_reports_zero() -> None:
    report = retrieval.evaluate([], k_values=(1, 5))
    assert report.recall == {1: 0.0, 5: 0.0}
    assert report.scored == 0


def test_summary_is_readable() -> None:
    report = retrieval.Report(recall={1: 0.5, 5: 0.875}, scored=8, skipped=2)
    assert report.summary() == "recall@1=0.500, recall@5=0.875  (n=8, skipped=2)"
