from pathlib import Path

import pytest

from askdb.agent.loop import Agent
from askdb.agent.tools import Toolbox
from askdb.db import catalog
from askdb.llm.types import Completion, Usage


class ScriptedModel:
    """Replays prepared responses and records the prompts it was given."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        variant: int = 0,
    ) -> Completion:
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else "SELECT 1"
        return Completion(
            text=text, model=model or "scripted", usage=Usage(100, 20), latency_ms=5.0
        )


@pytest.fixture
def toolbox(sample_db: Path) -> Toolbox:
    return Toolbox(sample_db, catalog.load(sample_db))


def test_answers_a_question_that_works_first_time(toolbox: Toolbox) -> None:
    model = ScriptedModel("```sql\nSELECT name FROM artist ORDER BY name\n```")
    result = Agent(toolbox, model).answer("list the artists")

    assert result.succeeded
    assert result.sql == "SELECT name FROM artist ORDER BY name"
    assert result.output is not None
    assert len(result.output.rows) == 3
    assert result.repairs == 0


def test_repairs_a_query_the_database_rejects(toolbox: Toolbox) -> None:
    model = ScriptedModel(
        "SELECT nope FROM artist",
        "SELECT name FROM artist",
    )
    result = Agent(toolbox, model).answer("list the artists")

    assert result.succeeded
    assert result.repairs == 1
    assert result.model_calls == 2


def test_the_repair_prompt_contains_the_database_error(toolbox: Toolbox) -> None:
    # This is the feedback edge the whole design rests on.
    model = ScriptedModel("SELECT nope FROM artist", "SELECT name FROM artist")
    Agent(toolbox, model).answer("list the artists")

    assert "no such column: nope" in model.prompts[1]
    assert "SELECT nope FROM artist" in model.prompts[1]


def test_gives_up_after_the_repair_budget(toolbox: Toolbox) -> None:
    model = ScriptedModel(*["SELECT nope FROM artist"] * 5)
    result = Agent(toolbox, model, max_repairs=2).answer("list the artists")

    assert not result.succeeded
    assert result.repairs == 2
    assert result.model_calls == 3


def test_zero_repairs_means_one_attempt(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT nope FROM artist", "SELECT name FROM artist")
    result = Agent(toolbox, model, max_repairs=0).answer("list the artists")

    assert not result.succeeded
    assert result.model_calls == 1


def test_stops_repairing_once_the_query_works(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT name FROM artist", "SELECT 999")
    result = Agent(toolbox, model, max_repairs=3).answer("list the artists")

    assert result.model_calls == 1
    assert model.prompts == model.prompts[:1]


def test_small_schemas_are_shown_whole(toolbox: Toolbox) -> None:
    # Three tables, five slots: retrieval could only lose information.
    agent = Agent(toolbox, ScriptedModel(), schema_tables=5)
    assert set(agent.select_tables("anything at all")) == {"artist", "album", "track"}


def test_large_schemas_are_narrowed_by_retrieval(toolbox: Toolbox) -> None:
    agent = Agent(toolbox, ScriptedModel(), schema_tables=1)
    assert agent.select_tables("which country is the musician from") == ("artist",)


def test_the_prompt_only_contains_the_selected_tables(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT name FROM artist")
    Agent(toolbox, model, schema_tables=1).answer("which country is the musician from")

    prompt = model.prompts[0]
    assert "CREATE TABLE artist" in prompt
    assert "CREATE TABLE track" not in prompt


def test_evidence_reaches_the_prompt(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT name FROM artist")
    Agent(toolbox, model).answer("list them", evidence="artists live in the artist table")

    assert "artists live in the artist table" in model.prompts[0]


def test_usage_is_summed_across_every_call(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT nope FROM artist", "SELECT name FROM artist")
    result = Agent(toolbox, model).answer("list the artists")

    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 40


def test_the_trace_records_every_step(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT nope FROM artist", "SELECT name FROM artist")
    result = Agent(toolbox, model).answer("list the artists")

    assert [step.kind for step in result.steps] == [
        "draft",
        "execute",
        "repair",
        "execute",
    ]


def test_execution_steps_carry_no_token_cost(toolbox: Toolbox) -> None:
    # Cost attribution has to separate model calls from database work.
    model = ScriptedModel("SELECT name FROM artist")
    result = Agent(toolbox, model).answer("list the artists")

    execute = next(step for step in result.steps if step.kind == "execute")
    assert execute.usage.total_tokens == 0


def test_example_values_are_off_by_default(toolbox: Toolbox) -> None:
    # Measured at +77% tokens for no accuracy change; see
    # docs/agent-experiments.md.
    model = ScriptedModel("SELECT name FROM artist")
    Agent(toolbox, model).answer("which country is the musician from")

    assert "e.g." not in model.prompts[0]
    assert "CREATE TABLE artist" in model.prompts[0]


def test_value_grounding_can_be_switched_on(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT name FROM artist")
    Agent(toolbox, model, ground_values=True).answer("which country")

    assert "e.g." in model.prompts[0]


def test_an_empty_result_triggers_a_recheck(toolbox: Toolbox) -> None:
    model = ScriptedModel(
        "SELECT name FROM artist WHERE country = 'XX'",
        "SELECT name FROM artist",
    )
    result = Agent(toolbox, model).answer("which artists are there")

    assert result.rechecks == 1
    assert result.concerns == ("empty_result",)
    assert result.output is not None
    assert len(result.output.rows) == 3


def test_the_recheck_prompt_explains_the_concern(toolbox: Toolbox) -> None:
    model = ScriptedModel(
        "SELECT name FROM artist WHERE country = 'XX'",
        "SELECT name FROM artist",
    )
    Agent(toolbox, model).answer("which artists are there")

    assert "returned no rows" in model.prompts[1]
    assert "executed without error" in model.prompts[1]


def test_a_clean_result_is_never_rechecked(toolbox: Toolbox) -> None:
    # Rechecking a healthy answer costs a call and risks making it worse.
    model = ScriptedModel("SELECT name FROM artist", "SELECT 999")
    result = Agent(toolbox, model).answer("which artists are there")

    assert result.rechecks == 0
    assert result.model_calls == 1


def test_only_one_recheck_happens(toolbox: Toolbox) -> None:
    model = ScriptedModel(*["SELECT name FROM artist WHERE country = 'XX'"] * 5)
    result = Agent(toolbox, model, max_repairs=3).answer("which artists are there")

    assert result.rechecks == 1
    assert result.model_calls == 2


def test_a_recheck_that_resolves_the_concern_is_adopted(toolbox: Toolbox) -> None:
    model = ScriptedModel(
        "SELECT name FROM artist WHERE country = 'XX'",
        "SELECT name FROM artist",
    )
    result = Agent(toolbox, model).answer("which artists are there")

    assert result.sql == "SELECT name FROM artist"


def test_a_recheck_that_stays_suspicious_keeps_the_first_answer(toolbox: Toolbox) -> None:
    # Neither result is convincing, so there is no reason to prefer the
    # second. Swapping would make the outcome depend on call ordering.
    model = ScriptedModel(
        "SELECT name FROM artist WHERE country = 'XX'",
        "SELECT name FROM artist WHERE country = 'YY'",
    )
    result = Agent(toolbox, model).answer("which artists are there")

    assert result.sql == "SELECT name FROM artist WHERE country = 'XX'"


def test_self_check_can_be_disabled_for_comparison(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT name FROM artist WHERE country = 'XX'")
    result = Agent(toolbox, model, self_check=False).answer("which artists are there")

    assert result.rechecks == 0
    assert result.model_calls == 1


def test_execution_errors_still_repair_rather_than_recheck(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT nope FROM artist", "SELECT name FROM artist")
    result = Agent(toolbox, model).answer("which artists are there")

    assert result.repairs == 1
    assert result.rechecks == 0


def test_a_single_sample_does_not_vote(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT name FROM artist")
    result = Agent(toolbox, model).answer("which artists are there")

    assert result.agreement is None
    assert result.model_calls == 1


def test_sampling_draws_the_query_several_times(toolbox: Toolbox) -> None:
    model = ScriptedModel(*["SELECT name FROM artist"] * 3)
    result = Agent(toolbox, model, samples=3).answer("which artists are there")

    assert result.model_calls == 3
    assert result.agreement == 1.0


def test_unanimous_draws_report_full_agreement(toolbox: Toolbox) -> None:
    model = ScriptedModel(
        "SELECT name FROM artist",
        "SELECT name FROM artist ORDER BY name DESC",
        "SELECT DISTINCT name FROM artist",
    )
    # Three different queries returning identical rows: that is agreement,
    # not confusion.
    result = Agent(toolbox, model, samples=3).answer("which artists are there")

    assert result.agreement == 1.0


def test_disagreeing_draws_lower_the_agreement(toolbox: Toolbox) -> None:
    model = ScriptedModel(
        "SELECT name FROM artist",
        "SELECT name FROM artist",
        "SELECT name FROM artist WHERE country = 'ML'",
    )
    result = Agent(toolbox, model, samples=3).answer("which artists are there")

    assert result.agreement == pytest.approx(2 / 3)


def test_disagreement_lowers_confidence(toolbox: Toolbox) -> None:
    model = ScriptedModel(
        "SELECT name FROM artist",
        "SELECT name FROM artist WHERE country = 'ML'",
        "SELECT name FROM artist WHERE country = 'US'",
    )
    result = Agent(toolbox, model, samples=3).answer("which artists are there")

    assert result.verdict is not None
    assert result.verdict.confidence < 1.0


def test_sampling_keeps_the_majority_answer(toolbox: Toolbox) -> None:
    model = ScriptedModel(
        "SELECT name FROM artist WHERE country = 'ML'",
        "SELECT name FROM artist",
        "SELECT name FROM artist",
    )
    result = Agent(toolbox, model, samples=3).answer("which artists are there")

    assert result.output is not None
    assert len(result.output.rows) == 3


def test_repairs_are_not_resampled(toolbox: Toolbox) -> None:
    # Three draws for the draft, then a single repair: a repair already has
    # the database's error to work from, which beats another opinion.
    model = ScriptedModel(
        *["SELECT nope FROM artist"] * 3,
        "SELECT name FROM artist",
    )
    result = Agent(toolbox, model, samples=3).answer("which artists are there")

    assert result.model_calls == 4
    assert result.repairs == 1


def test_sampling_records_a_vote_step(toolbox: Toolbox) -> None:
    model = ScriptedModel(*["SELECT name FROM artist"] * 2)
    result = Agent(toolbox, model, samples=2).answer("which artists are there")

    vote_step = next(step for step in result.steps if step.kind == "vote")
    assert "agreement" in vote_step.detail


def test_tables_shown_are_reported(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT name FROM artist")
    result = Agent(toolbox, model, schema_tables=1).answer("which country")

    assert result.tables_shown == ("artist",)
