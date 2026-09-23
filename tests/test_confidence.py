from askdb.agent import confidence

SCHEMA = confidence.schema_vocabulary(
    ["artist", "album"], ["artist_id", "name", "country", "title", "released"]
)


def assess(question: str, **kwargs: object) -> confidence.Verdict:
    return confidence.assess(question, schema_tokens=SCHEMA, **kwargs)  # type: ignore[arg-type]


def test_a_clean_answer_is_offered() -> None:
    verdict = assess("which artists are from the country Mali")

    assert verdict.decision == confidence.ANSWER
    assert verdict.confidence == 1.0
    assert verdict.reasons == ()


def test_a_failed_query_is_refused_outright() -> None:
    verdict = assess("which artists", query_failed=True)

    assert verdict.decision == confidence.REFUSE
    assert verdict.confidence == 0.0


def test_an_unresolved_concern_lowers_confidence() -> None:
    verdict = assess("which artists are from the country Mali", unresolved_concern=True)

    assert verdict.confidence < 1.0
    assert any("still looked wrong" in reason for reason in verdict.reasons)


def test_repairs_lower_confidence_in_proportion() -> None:
    one = assess("which artists are from the country Mali", repairs=1)
    two = assess("which artists are from the country Mali", repairs=2)

    assert two.confidence < one.confidence < 1.0


def test_a_question_about_absent_concepts_loses_confidence() -> None:
    # Nothing in this schema stores weather.
    verdict = assess("what was the rainfall and humidity during the thunderstorm")

    assert verdict.confidence < 1.0
    assert any("appear anywhere in the schema" in reason for reason in verdict.reasons)


def test_a_well_covered_question_keeps_full_confidence() -> None:
    assert assess("list the album title and released year").confidence == 1.0


def test_stopwords_do_not_count_against_coverage() -> None:
    # Otherwise every ordinary English question would look uncovered.
    assert confidence.vocabulary_coverage("how many of the artists are there", SCHEMA) == 1.0


def test_a_question_with_no_content_words_is_fully_covered() -> None:
    assert confidence.vocabulary_coverage("how many are there", SCHEMA) == 1.0


def test_accumulated_doubt_triggers_a_clarification() -> None:
    verdict = assess(
        "which artists are from the country Mali", unresolved_concern=True, repairs=1
    )

    assert verdict.decision == confidence.CLARIFY
    assert len(verdict.reasons) == 2


def test_enough_doubt_triggers_a_refusal() -> None:
    verdict = assess(
        "what was the rainfall during the thunderstorm",
        unresolved_concern=True,
        repairs=2,
    )

    assert verdict.decision == confidence.REFUSE


def test_confidence_never_goes_below_zero() -> None:
    verdict = assess(
        "what was the rainfall during the thunderstorm",
        unresolved_concern=True,
        repairs=99,
    )
    assert verdict.confidence == 0.0


def test_thresholds_are_tunable() -> None:
    # The sweep that produces the coverage curve moves exactly these.
    question = "which artists are from the country Mali"

    strict = assess(question, repairs=1, clarify_below=0.95)
    lenient = assess(question, repairs=1, clarify_below=0.5)

    assert strict.decision == confidence.CLARIFY
    assert lenient.decision == confidence.ANSWER


def test_schema_vocabulary_splits_compound_names() -> None:
    # Split on both underscore and camel case, then stemmed.
    vocabulary = confidence.schema_vocabulary(["sat_scores"], ["CDSCode"])
    assert {"sat", "score", "cds", "code"} <= vocabulary


def test_content_tokens_drop_stopwords() -> None:
    assert confidence.content_tokens("how many of the artists") == ["artist"]


def test_plurals_match_singular_schema_names() -> None:
    # Questions are plural, schemas are singular. Without stemming, "which
    # artists" scores zero coverage against a table named artist.
    assert confidence.vocabulary_coverage("which artists", SCHEMA) == 1.0


def test_singular_questions_match_plural_schema_names() -> None:
    vocabulary = confidence.schema_vocabulary(["schools"], ["county"])
    assert confidence.vocabulary_coverage("which school is in the county", vocabulary) == 1.0


def test_stemming_handles_common_endings() -> None:
    assert confidence.stem("countries") == "country"
    assert confidence.stem("boxes") == "box"
    assert confidence.stem("matches") == "match"
    assert confidence.stem("schools") == "school"


def test_stemming_leaves_short_and_double_s_words_alone() -> None:
    # "class" and "is" must not become "clas" and "i".
    assert confidence.stem("class") == "class"
    assert confidence.stem("is") == "is"
    assert confidence.stem("sat") == "sat"
