from askdb.agent import prompts


def test_extracts_sql_from_a_tagged_fence() -> None:
    response = "```sql\nSELECT 1\n```"
    assert prompts.extract_sql(response) == "SELECT 1"


def test_extracts_sql_from_an_untagged_fence() -> None:
    assert prompts.extract_sql("```\nSELECT 1\n```") == "SELECT 1"


def test_ignores_commentary_around_the_fence() -> None:
    # Models add a sentence despite being told not to.
    response = "Here is the query you asked for:\n\n```sql\nSELECT 1\n```\n\nHope that helps!"
    assert prompts.extract_sql(response) == "SELECT 1"


def test_accepts_bare_sql() -> None:
    assert prompts.extract_sql("  SELECT 1  ") == "SELECT 1"


def test_takes_the_first_fence_when_several_appear() -> None:
    response = "```sql\nSELECT 1\n```\nor maybe\n```sql\nSELECT 2\n```"
    assert prompts.extract_sql(response) == "SELECT 1"


def test_preserves_multiline_queries() -> None:
    response = "```sql\nSELECT a\nFROM t\nWHERE b = 1\n```"
    assert prompts.extract_sql(response) == "SELECT a\nFROM t\nWHERE b = 1"


def test_fence_tag_is_case_insensitive() -> None:
    assert prompts.extract_sql("```SQL\nSELECT 1\n```") == "SELECT 1"


def test_draft_prompt_carries_schema_and_question() -> None:
    prompt = prompts.draft_prompt("CREATE TABLE t (a INT);", "how many?")
    assert "CREATE TABLE t (a INT);" in prompt
    assert "Question: how many?" in prompt


def test_draft_prompt_includes_evidence_when_present() -> None:
    prompt = prompts.draft_prompt("schema", "q", "count means COUNT(*)")
    assert "Hint: count means COUNT(*)" in prompt


def test_draft_prompt_omits_an_empty_hint_label() -> None:
    # A bare "Hint:" with nothing after it reads as a missing value.
    assert "Hint:" not in prompts.draft_prompt("schema", "q", "   ")


def test_repair_prompt_carries_the_failed_query_and_error() -> None:
    prompt = prompts.repair_prompt(
        "schema", "q", "SELECT nope FROM t", "no such column: nope"
    )
    assert "SELECT nope FROM t" in prompt
    assert "no such column: nope" in prompt
