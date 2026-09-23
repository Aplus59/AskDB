import pytest

from askdb.agent.consistency import Consensus, Draw, signature, vote
from askdb.agent.tools import QueryOutput


def rows(*values: object) -> QueryOutput:
    return QueryOutput(columns=("n",), rows=tuple((v,) for v in values))


def failed(message: str = "no such column: nope") -> QueryOutput:
    return QueryOutput(columns=(), rows=(), error=message)


def draw(sql: str, output: QueryOutput) -> Draw:
    return Draw(sql=sql, output=output)


def test_unanimous_draws_score_full_agreement() -> None:
    result = vote([draw("SELECT 1", rows(1)) for _ in range(5)])

    assert result.agreement == 1.0
    assert result.unanimous
    assert result.distinct_answers == 1


def test_agreement_is_the_share_landing_on_the_winner() -> None:
    draws = [
        draw("a", rows(1)),
        draw("b", rows(1)),
        draw("c", rows(1)),
        draw("d", rows(2)),
    ]
    result = vote(draws)

    assert result.agreement == 0.75
    assert result.distinct_answers == 2
    assert result.output.rows == ((1,),)


def test_different_sql_returning_the_same_rows_counts_as_agreement() -> None:
    # The point of comparing results rather than text: these are the same
    # answer written two ways.
    draws = [
        draw("SELECT name FROM artist ORDER BY name", rows("a", "b")),
        draw("SELECT DISTINCT name FROM artist", rows("b", "a")),
    ]
    result = vote(draws)

    assert result.agreement == 1.0
    assert result.unanimous


def test_total_disagreement_scores_the_floor() -> None:
    draws = [draw("a", rows(1)), draw("b", rows(2)), draw("c", rows(3))]
    result = vote(draws)

    assert result.agreement == pytest.approx(1 / 3)
    assert result.distinct_answers == 3


def test_failures_count_as_a_vote_against_confidence() -> None:
    # Dropping failed draws would report high agreement for a question the
    # model kept getting wrong.
    draws = [draw("a", rows(1)), draw("b", failed()), draw("c", failed())]
    result = vote(draws)

    assert result.agreement == pytest.approx(2 / 3)
    assert result.distinct_answers == 2


def test_all_failures_still_returns_a_draw() -> None:
    draws = [draw("a", failed()), draw("b", failed())]
    result = vote(draws)

    assert result.agreement == 1.0
    assert not result.output.ok


def test_a_working_query_is_preferred_over_a_winning_failure() -> None:
    # Two failures outvote one success, but returning the failure would throw
    # away the only usable answer.
    draws = [draw("a", failed()), draw("b", failed()), draw("c", rows(7))]
    result = vote(draws)

    assert result.output.ok
    assert result.sql == "c"
    assert result.agreement == pytest.approx(2 / 3)


def test_different_errors_are_not_treated_as_different_answers() -> None:
    # Two ways of failing are not two opinions.
    draws = [draw("a", failed("no such column: x")), draw("b", failed("syntax error"))]
    assert vote(draws).distinct_answers == 1


def test_ties_resolve_to_the_earliest_draw() -> None:
    # Determinism: two runs over the same draws must agree.
    draws = [draw("first", rows(1)), draw("second", rows(2))]
    assert vote(draws).sql == "first"


def test_a_single_draw_is_trivially_unanimous() -> None:
    result = vote([draw("only", rows(1))])

    assert result.agreement == 1.0
    assert result.draws == 1


def test_voting_on_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="cannot vote on nothing"):
        vote([])


def test_signature_ignores_row_order() -> None:
    assert signature(rows(1, 2)) == signature(rows(2, 1))


def test_signature_separates_different_results() -> None:
    assert signature(rows(1)) != signature(rows(2))


def test_consensus_reports_how_many_draws_it_saw() -> None:
    result = vote([draw("a", rows(1)) for _ in range(4)])
    assert isinstance(result, Consensus)
    assert result.draws == 4
