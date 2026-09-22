from pathlib import Path

import pytest

from askdb.eval import audit
from askdb.eval.run import QuestionResult


def a_result(question_id: int, *, match: bool) -> QuestionResult:
    return QuestionResult(
        question_id=question_id,
        db_id="toxicology",
        question="how many?",
        gold_sql="SELECT 1",
        predicted_sql="SELECT 2",
        match=match,
        exact_match=match,
        failure_reason=None if match else "wrong_rows",
        repairs=0,
        model_calls=1,
        input_tokens=100,
        output_tokens=20,
        latency_ms=10.0,
        tables_shown=("molecule",),
    )


def test_rejects_an_unknown_category() -> None:
    with pytest.raises(audit.UnknownCategory, match="not one of"):
        audit.Label(question_id=1, category="just_wrong")


def test_accepts_every_documented_category() -> None:
    for category in audit.CATEGORIES:
        assert audit.Label(question_id=1, category=category).category == category


def test_saves_and_reloads_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    audit.save_label(path, audit.Label(1, audit.MODEL_ERROR, "bad join"))

    loaded = audit.load_labels(path)
    assert loaded[1].category == audit.MODEL_ERROR
    assert loaded[1].note == "bad join"


def test_relabelling_overrides_the_earlier_judgement(tmp_path: Path) -> None:
    # Re-reading a question after seeing its rows often changes the verdict.
    path = tmp_path / "labels.jsonl"
    audit.save_label(path, audit.Label(1, audit.MODEL_ERROR))
    audit.save_label(path, audit.Label(1, audit.FORMAT_MISMATCH))

    assert audit.load_labels(path)[1].category == audit.FORMAT_MISMATCH


def test_corrupt_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    audit.save_label(path, audit.Label(1, audit.MODEL_ERROR))
    path.write_text(path.read_text(encoding="utf-8") + '{"broken', encoding="utf-8")

    assert set(audit.load_labels(path)) == {1}


def test_missing_label_file_is_empty(tmp_path: Path) -> None:
    assert audit.load_labels(tmp_path / "absent.jsonl") == {}


def test_counts_failures_and_labels() -> None:
    results = [a_result(1, match=True), a_result(2, match=False), a_result(3, match=False)]
    labels = {2: audit.Label(2, audit.MODEL_ERROR)}

    summary = audit.summarize(results, labels)
    assert summary.total == 3
    assert summary.matched == 1
    assert summary.failures == 2
    assert summary.labelled == 1
    assert summary.unlabelled == 1


def test_raw_accuracy_is_unadjusted() -> None:
    results = [a_result(1, match=True), a_result(2, match=False)]
    assert audit.summarize(results, {}).raw_accuracy == 0.5


def test_annotation_errors_leave_the_denominator() -> None:
    # Three questions, one correct, one a real miss, one the benchmark got
    # wrong. Corrected accuracy is 1 of 2, not 1 of 3.
    results = [a_result(i, match=(i == 1)) for i in (1, 2, 3)]
    labels = {
        2: audit.Label(2, audit.MODEL_ERROR),
        3: audit.Label(3, audit.ANNOTATION_ERROR),
    }

    summary = audit.summarize(results, labels)
    assert summary.raw_accuracy == pytest.approx(1 / 3)
    assert summary.corrected_accuracy == 0.5


def test_unanswerable_questions_also_leave_the_denominator() -> None:
    results = [a_result(1, match=True), a_result(2, match=False)]
    labels = {2: audit.Label(2, audit.SCHEMA_LIMIT)}

    assert audit.summarize(results, labels).corrected_accuracy == 1.0


def test_an_annotation_error_is_not_counted_as_correct() -> None:
    # Removing it from the denominator is honest; calling it a success is not.
    results = [a_result(1, match=False)]
    labels = {1: audit.Label(1, audit.ANNOTATION_ERROR)}

    summary = audit.summarize(results, labels)
    assert summary.matched == 0
    assert summary.corrected_accuracy == 0.0


def test_semantic_accuracy_includes_shape_mismatches() -> None:
    results = [a_result(1, match=True), a_result(2, match=False), a_result(3, match=False)]
    labels = {
        2: audit.Label(2, audit.FORMAT_MISMATCH),
        3: audit.Label(3, audit.MODEL_ERROR),
    }

    summary = audit.summarize(results, labels)
    assert summary.raw_accuracy == pytest.approx(1 / 3)
    assert summary.semantic_accuracy == pytest.approx(2 / 3)


def test_format_mismatches_stay_in_the_corrected_denominator() -> None:
    # The reasoning was right, but the user still got the wrong shape back.
    results = [a_result(1, match=False)]
    labels = {1: audit.Label(1, audit.FORMAT_MISMATCH)}

    assert audit.summarize(results, labels).corrected_accuracy == 0.0


def test_summary_of_nothing_is_zeroed() -> None:
    summary = audit.summarize([], {})
    assert summary.raw_accuracy == 0.0
    assert summary.corrected_accuracy == 0.0
    assert summary.semantic_accuracy == 0.0


def test_render_reports_the_three_numbers_and_the_taxonomy() -> None:
    results = [a_result(1, match=True), a_result(2, match=False)]
    labels = {2: audit.Label(2, audit.ANNOTATION_ERROR)}

    rendered = audit.summarize(results, labels).render()
    assert "raw accuracy        0.500" in rendered
    assert "corrected accuracy  1.000" in rendered
    assert "annotation_error" in rendered


def test_render_flags_unreviewed_failures() -> None:
    results = [a_result(1, match=False)]
    assert "1 failures still unlabelled" in audit.summarize(results, {}).render()
