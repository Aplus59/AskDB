from askdb.agent import confidence
from askdb.eval import abstention
from askdb.eval.run import QuestionResult


def a_result(
    question_id: int,
    *,
    match: bool,
    repairs: int = 0,
    unresolved_concern: bool = False,
    coverage: float = 1.0,
    agreement: float | None = None,
    reason: str | None = None,
) -> QuestionResult:
    return QuestionResult(
        question_id=question_id,
        db_id="toxicology",
        question="how many?",
        gold_sql="SELECT 1",
        predicted_sql="SELECT 1",
        match=match,
        exact_match=match,
        failure_reason=reason if reason else (None if match else "wrong_rows"),
        repairs=repairs,
        model_calls=1,
        input_tokens=10,
        output_tokens=2,
        latency_ms=1.0,
        tables_shown=("molecule",),
        unresolved_concern=unresolved_concern,
        coverage=coverage,
        agreement=agreement,
    )


def test_a_clean_result_is_answered() -> None:
    decision = abstention.decide(a_result(1, match=True), 0.6, 0.3)
    assert decision == confidence.ANSWER


def test_a_query_that_never_ran_is_refused() -> None:
    result = a_result(1, match=False, reason="execution_error")
    assert abstention.decide(result, 0.0, 0.0) == confidence.REFUSE


def test_repairs_do_not_lower_the_decision() -> None:
    # Measured lift of 0.84x: a repaired query is no likelier to be wrong,
    # because the database rejected the first attempt and the model fixed it.
    result = a_result(1, match=True, repairs=2)
    assert abstention.decide(result, 0.95, 0.3) == confidence.ANSWER


def test_low_vocabulary_coverage_does_not_lower_the_decision() -> None:
    # Measured lift of 1.05x, which is the base rate with noise.
    result = a_result(1, match=True, coverage=0.0)
    assert abstention.decide(result, 0.95, 0.3) == confidence.ANSWER


def test_an_unresolved_concern_lowers_the_decision() -> None:
    result = a_result(1, match=True, unresolved_concern=True)
    assert abstention.decide(result, 0.7, 0.3) == confidence.CLARIFY


def test_enough_doubt_refuses() -> None:
    result = a_result(1, match=True, unresolved_concern=True, agreement=0.34)
    assert abstention.decide(result, 0.7, 0.3) == confidence.REFUSE


def test_answering_everything_gives_full_coverage() -> None:
    results = [a_result(1, match=True), a_result(2, match=False)]
    point = abstention.evaluate_at(results, 0.0, 0.0)

    assert point.coverage == 1.0
    assert point.accuracy_when_answered == 0.5
    assert point.silent_error_rate == 0.5


def test_answering_nothing_gives_zero_coverage() -> None:
    results = [a_result(1, match=True), a_result(2, match=False)]
    point = abstention.evaluate_at(results, 1.01, 0.0)

    assert point.coverage == 0.0
    assert point.silent_error_rate == 0.0
    assert point.abstained == 2


def test_silent_errors_count_only_answered_and_wrong() -> None:
    # A wrong answer the system declined to give is not a silent error.
    results = [
        a_result(1, match=False),
        a_result(2, match=False, unresolved_concern=True, agreement=0.34),
    ]
    point = abstention.evaluate_at(results, 0.6, 0.3)

    assert point.silent_errors == 1
    assert point.abstained_wrong == 1


def test_abstention_precision_rewards_declining_wrong_answers() -> None:
    results = [
        a_result(1, match=True),
        a_result(2, match=False, unresolved_concern=True, agreement=0.34),
    ]
    point = abstention.evaluate_at(results, 0.6, 0.3)

    assert point.abstained == 1
    assert point.abstention_precision == 1.0


def test_abstention_precision_punishes_declining_good_answers() -> None:
    results = [
        a_result(1, match=True, unresolved_concern=True, agreement=0.34),
        a_result(2, match=True),
    ]
    point = abstention.evaluate_at(results, 0.6, 0.3)

    assert point.abstained == 1
    assert point.abstention_precision == 0.0


def test_accuracy_rises_as_coverage_falls() -> None:
    # The trade the whole exercise is about.
    results = [
        a_result(1, match=True),
        a_result(2, match=False, unresolved_concern=True),
    ]
    lenient = abstention.evaluate_at(results, 0.0, 0.0)
    strict = abstention.evaluate_at(results, 0.7, 0.3)

    assert lenient.coverage > strict.coverage
    assert strict.accuracy_when_answered > lenient.accuracy_when_answered


def test_sweep_returns_one_point_per_threshold() -> None:
    results = [a_result(1, match=True)]
    assert len(abstention.sweep(results, thresholds=(0.0, 0.5, 1.01))) == 3


def test_empty_results_do_not_divide_by_zero() -> None:
    point = abstention.evaluate_at([], 0.6, 0.3)

    assert point.coverage == 0.0
    assert point.accuracy_when_answered == 0.0
    assert point.silent_error_rate == 0.0
    assert point.abstention_precision == 0.0


def test_render_includes_a_row_per_point() -> None:
    results = [a_result(1, match=True)]
    rendered = abstention.render(abstention.sweep(results, thresholds=(0.0, 1.01)))

    assert len(rendered.splitlines()) == 4  # header, rule, two rows
