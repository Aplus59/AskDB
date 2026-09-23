from pathlib import Path

from askdb.data.minidev import Question
from askdb.eval import run


def a_result(
    question_id: int = 1,
    *,
    match: bool = True,
    exact_match: bool = True,
    repairs: int = 0,
    reason: str | None = None,
    difficulty: str | None = "simple",
    tokens: tuple[int, int] = (100, 20),
    calls: int = 1,
) -> run.QuestionResult:
    return run.QuestionResult(
        question_id=question_id,
        db_id="toxicology",
        question="how many?",
        gold_sql="SELECT 1",
        predicted_sql="SELECT 1",
        match=match,
        exact_match=exact_match,
        failure_reason=reason,
        repairs=repairs,
        model_calls=calls,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        latency_ms=12.0,
        tables_shown=("molecule",),
        difficulty=difficulty,
    )


def test_round_trips_through_json() -> None:
    original = a_result()
    restored = run.QuestionResult.from_json(original.to_json())
    assert restored == original


def test_tables_shown_survives_as_a_tuple() -> None:
    restored = run.QuestionResult.from_json(a_result().to_json())
    assert restored.tables_shown == ("molecule",)


def test_appends_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    run.append_result(path, a_result(1))
    run.append_result(path, a_result(2))

    loaded = run.load_results(path)
    assert set(loaded) == {1, 2}


def test_reloading_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert run.load_results(tmp_path / "absent.jsonl") == {}


def test_a_truncated_final_line_does_not_discard_the_run(tmp_path: Path) -> None:
    # A run killed mid-write leaves a partial line. Losing 400 completed
    # questions because of it would be far worse than skipping one.
    path = tmp_path / "run.jsonl"
    run.append_result(path, a_result(1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"question_id": 2, "db_id": "tox')

    assert set(run.load_results(path)) == {1}


def test_a_later_line_replaces_an_earlier_one(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    run.append_result(path, a_result(1, match=False))
    run.append_result(path, a_result(1, match=True))

    loaded = run.load_results(path)
    assert loaded[1].match


def test_pending_skips_what_is_done() -> None:
    questions = [
        Question(question_id=i, db_id="toxicology", question="q", gold_sql="SELECT 1")
        for i in (1, 2, 3)
    ]
    done = {2: a_result(2)}

    assert [q.question_id for q in run.pending(questions, done)] == [1, 3]


def questions_across(*counts: tuple[str, int]) -> list[Question]:
    made: list[Question] = []
    next_id = 0
    for db_id, count in counts:
        for _ in range(count):
            made.append(
                Question(question_id=next_id, db_id=db_id, question="q", gold_sql="SELECT 1")
            )
            next_id += 1
    return made


def test_sampling_spreads_across_databases() -> None:
    # Taking the first N would draw all six from "alpha" and measure one schema.
    questions = questions_across(("alpha", 10), ("beta", 10), ("gamma", 10))
    chosen = run.stratified_sample(questions, 6)

    counts = {db: sum(1 for q in chosen if q.db_id == db) for db in ("alpha", "beta", "gamma")}
    assert counts == {"alpha": 2, "beta": 2, "gamma": 2}


def test_sampling_is_deterministic() -> None:
    questions = questions_across(("alpha", 5), ("beta", 5))
    assert run.stratified_sample(questions, 4) == run.stratified_sample(questions, 4)


def test_sampling_drains_uneven_databases() -> None:
    questions = questions_across(("alpha", 1), ("beta", 5))
    chosen = run.stratified_sample(questions, 4)

    assert len(chosen) == 4
    assert sum(1 for q in chosen if q.db_id == "alpha") == 1


def test_sampling_cannot_exceed_what_exists() -> None:
    assert len(run.stratified_sample(questions_across(("alpha", 3)), 99)) == 3


def test_sampling_nothing_returns_nothing() -> None:
    assert run.stratified_sample(questions_across(("alpha", 3)), 0) == []


def test_summary_of_nothing_is_zeroed() -> None:
    summary = run.summarize([])
    assert summary.total == 0
    assert summary.accuracy == 0.0


def test_accuracy_counts_matches() -> None:
    results = [a_result(1), a_result(2, match=False, reason="wrong_rows"), a_result(3)]
    assert run.summarize(results).accuracy == 2 / 3


def test_strict_accuracy_is_reported_separately() -> None:
    results = [a_result(1, exact_match=False), a_result(2)]
    summary = run.summarize(results)

    assert summary.accuracy == 1.0
    assert summary.strict_accuracy == 0.5


def test_repair_rate_counts_questions_that_needed_one() -> None:
    results = [a_result(1, repairs=1), a_result(2), a_result(3, repairs=2)]
    assert run.summarize(results).repair_rate == 2 / 3


def test_recovery_rate_counts_only_repairs_that_worked() -> None:
    # This is the number that says whether the repair loop earns its cost.
    results = [
        a_result(1, repairs=1, match=True),
        a_result(2, repairs=1, match=False, reason="execution_error"),
        a_result(3, repairs=0, match=True),
    ]
    assert run.summarize(results).recovery_rate == 1 / 3


def test_failures_are_grouped_by_reason() -> None:
    results = [
        a_result(1, match=False, reason="wrong_rows"),
        a_result(2, match=False, reason="wrong_rows"),
        a_result(3, match=False, reason="timeout"),
        a_result(4),
    ]
    assert run.summarize(results).failures == {"wrong_rows": 2, "timeout": 1}


def test_a_correct_answer_contributes_no_failure_reason() -> None:
    assert run.summarize([a_result(1, reason="wrong_rows")]).failures == {}


def test_accuracy_is_broken_down_by_difficulty() -> None:
    results = [
        a_result(1, difficulty="simple"),
        a_result(2, difficulty="simple", match=False, reason="wrong_rows"),
        a_result(3, difficulty="challenging"),
    ]
    by_difficulty = run.summarize(results).by_difficulty

    assert by_difficulty["simple"] == (2, 0.5)
    assert by_difficulty["challenging"] == (1, 1.0)


def test_averages_tokens_and_calls() -> None:
    results = [a_result(1, tokens=(100, 20), calls=1), a_result(2, tokens=(200, 40), calls=3)]
    summary = run.summarize(results)

    assert summary.avg_tokens == 180
    assert summary.avg_model_calls == 2


def test_summary_renders_the_headline_numbers() -> None:
    rendered = run.summarize([a_result(1), a_result(2, match=False, reason="wrong_rows")]).render()

    assert "execution accuracy 0.500" in rendered
    assert "wrong_rows" in rendered


def test_a_question_whose_reference_fails_is_not_counted_against_accuracy() -> None:
    # Two of the 500 Mini-Dev references time out even given thirty seconds.
    # Counting those as failures blames the system for the benchmark.
    results = [
        a_result(1, match=True),
        a_result(2, match=False, reason="gold_failed"),
    ]
    summary = run.summarize(results)

    assert summary.total == 1
    assert summary.unscorable == 1
    assert summary.accuracy == 1.0


def test_unscorable_questions_are_reported_not_hidden() -> None:
    results = [a_result(1, match=True), a_result(2, match=False, reason="gold_failed")]
    assert "unscorable         1" in run.summarize(results).render()


def test_unscorable_questions_leave_the_failure_breakdown() -> None:
    results = [
        a_result(1, match=False, reason="wrong_rows"),
        a_result(2, match=False, reason="gold_failed"),
    ]
    assert run.summarize(results).failures == {"wrong_rows": 1}


def test_everything_unscorable_still_reports_the_count() -> None:
    summary = run.summarize([a_result(1, match=False, reason="gold_failed")])

    assert summary.total == 0
    assert summary.unscorable == 1


def test_no_unscorable_questions_means_no_extra_line() -> None:
    assert "unscorable" not in run.summarize([a_result(1, match=True)]).render()
