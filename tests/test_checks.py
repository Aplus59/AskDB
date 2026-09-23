import pytest

from askdb.agent import checks
from askdb.agent.tools import QueryOutput


def output(rows: tuple[tuple[object, ...], ...], columns: tuple[str, ...] = ("n",)) -> QueryOutput:
    return QueryOutput(columns=columns, rows=rows)


def failed() -> QueryOutput:
    return QueryOutput(columns=(), rows=(), error="no such column: nope")


def test_a_normal_result_raises_nothing() -> None:
    assert checks.inspect("how many artists are there", output(((5,),))) is None


def test_empty_result_is_suspicious() -> None:
    # The measured case: a filter spelled 'East Bohemia' against data storing
    # 'east Bohemia'. Valid SQL, no error, zero rows.
    suspicion = checks.inspect("how many accounts are in east Bohemia", output(()))

    assert suspicion is not None
    assert suspicion.code == checks.EMPTY_RESULT
    assert "capitalised" in suspicion.message


def test_a_single_null_is_suspicious() -> None:
    suspicion = checks.inspect("what is the average score", output(((None,),)))

    assert suspicion is not None
    assert suspicion.code == checks.NULL_SCALAR


def test_a_scalar_question_returning_many_rows_is_suspicious() -> None:
    suspicion = checks.inspect("how many schools are there", output(((1,), (2,), (3,))))

    assert suspicion is not None
    assert suspicion.code == checks.EXPECTED_ONE_ROW
    assert "3 rows" in suspicion.message


def test_a_list_question_returning_many_rows_is_fine() -> None:
    # The common case. Firing here would spend a model call on every healthy
    # listing query.
    assert checks.inspect("which schools are in Alameda", output(((1,), (2,)))) is None


def test_a_failed_query_is_left_to_the_repair_loop() -> None:
    assert checks.inspect("how many", failed()) is None


def test_a_legitimate_zero_count_is_not_flagged() -> None:
    # COUNT over no matching rows still returns one row containing 0.
    assert checks.inspect("how many artists are from Mars", output(((0,),))) is None


def test_null_inside_a_multi_column_row_is_not_flagged() -> None:
    # Only a lone NULL scalar is evidence of a filter matching nothing.
    assert checks.inspect("what is the average", output(((None, 5),), ("a", "b"))) is None


def test_null_among_several_rows_is_not_flagged() -> None:
    assert checks.inspect("list the values", output(((None,), (2,)))) is None


@pytest.mark.parametrize(
    "question",
    [
        "How many schools are there?",
        "how much did it cost",
        "What is the total number of accounts?",
        "What is the average score in maths?",
        "Calculate the average number of oxygen atoms.",
        "Count the cards with foils.",
    ],
)
def test_scalar_questions_are_recognised(question: str) -> None:
    assert checks.expects_single_value(question)


@pytest.mark.parametrize(
    "question",
    [
        "Which schools have more than 500 students?",
        "List the codes of the schools.",
        "Name the artists from Mali.",
        "Show me every account that has how many transactions recorded",
    ],
)
def test_listing_questions_are_not_treated_as_scalar(question: str) -> None:
    assert not checks.expects_single_value(question)
