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
        self, prompt: str, *, model: str | None = None, temperature: float = 0.0
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


def test_tables_shown_are_reported(toolbox: Toolbox) -> None:
    model = ScriptedModel("SELECT name FROM artist")
    result = Agent(toolbox, model, schema_tables=1).answer("which country")

    assert result.tables_shown == ("artist",)
