from askdb.eval import cost
from askdb.eval.run import QuestionResult


def a_result(
    question_id: int = 1,
    *,
    model: str | None = "gemini-3.5-flash-lite",
    tokens: tuple[int, int] = (1000, 100),
) -> QuestionResult:
    return QuestionResult(
        question_id=question_id,
        db_id="toxicology",
        question="how many?",
        gold_sql="SELECT 1",
        predicted_sql="SELECT 1",
        match=True,
        exact_match=True,
        failure_reason=None,
        repairs=0,
        model_calls=1,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        latency_ms=10.0,
        tables_shown=("molecule",),
        model=model,
    )


def test_prices_a_known_model() -> None:
    # 1000 input at $0.30/M = $0.0003; 100 output at $2.50/M = $0.00025.
    report = cost.estimate([a_result()])

    assert report.priced_questions == 1
    assert report.usd == 0.00055


def test_scales_per_thousand_questions() -> None:
    report = cost.estimate([a_result(i) for i in range(10)])

    assert report.usd_per_question == 0.00055
    assert report.usd_per_thousand == 0.55


def test_an_unknown_model_is_not_guessed() -> None:
    # A plausible-looking cost from a guessed rate is worse than a gap,
    # because nobody checks a number that looks reasonable.
    report = cost.estimate([a_result(model="some-future-model")])

    assert report.priced_questions == 0
    assert report.unpriced_questions == 1
    assert report.usd == 0.0
    assert report.unpriced_models == ("some-future-model",)


def test_a_missing_model_is_unpriced() -> None:
    report = cost.estimate([a_result(model=None)])

    assert report.unpriced_questions == 1
    assert report.unpriced_models == ("unknown",)


def test_priced_and_unpriced_questions_coexist() -> None:
    report = cost.estimate([a_result(1), a_result(2, model="mystery")])

    assert report.priced_questions == 1
    assert report.unpriced_questions == 1
    assert report.total_questions == 2


def test_unpriced_questions_do_not_dilute_the_per_question_cost() -> None:
    # Dividing by questions we could not price would understate the real rate.
    report = cost.estimate([a_result(1), a_result(2, model="mystery")])
    assert report.usd_per_question == 0.00055


def test_output_tokens_cost_more_than_input() -> None:
    heavy_input = cost.estimate([a_result(tokens=(10_000, 0))])
    heavy_output = cost.estimate([a_result(tokens=(0, 10_000))])

    assert heavy_output.usd > heavy_input.usd


def test_nothing_costs_nothing() -> None:
    report = cost.estimate([])

    assert report.usd == 0.0
    assert report.usd_per_question == 0.0
    assert report.total_questions == 0


def test_render_states_when_prices_were_checked() -> None:
    rendered = cost.estimate([a_result()]).render()

    assert cost.PRICES_CHECKED in rendered
    assert "per 1000 questions" in rendered


def test_render_names_unpriced_models() -> None:
    rendered = cost.estimate([a_result(1, model="mystery")]).render()
    assert "mystery" in rendered


def test_an_assumed_model_prices_older_runs() -> None:
    # Runs made before the model field existed can still be costed, but only
    # when the caller says which model to assume.
    report = cost.estimate([a_result(model=None)], assume_model="gemini-3.5-flash-lite")

    assert report.priced_questions == 1
    assert report.assumed_questions == 1
    assert report.usd == 0.00055


def test_the_assumption_is_named_in_the_output() -> None:
    # The assumption has to travel with the number, not disappear into it.
    rendered = cost.estimate(
        [a_result(model=None)], assume_model="gemini-3.5-flash-lite"
    ).render()

    assert "assumed gemini-3.5-flash-lite" in rendered


def test_recorded_models_are_not_overridden_by_the_assumption() -> None:
    report = cost.estimate(
        [a_result(model="gemini-2.5-flash-lite")], assume_model="gemini-3.5-flash-lite"
    )

    assert report.assumed_questions == 0
    # Priced at 2.5-flash-lite rates, not the assumed model's.
    assert report.usd == (1000 * 0.10 + 100 * 0.40) / 1_000_000


def test_an_unknown_assumed_model_is_still_unpriced() -> None:
    report = cost.estimate([a_result(model=None)], assume_model="not-a-model")

    assert report.priced_questions == 0
    assert report.unpriced_models == ("not-a-model",)
