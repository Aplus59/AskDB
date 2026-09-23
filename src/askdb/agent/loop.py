"""Draft, execute, read the real error, repair.

The repair step is the reason this is an agent rather than a prompt. A model
writing SQL blind gets column names and joins wrong; the same model shown the
database's own error message usually fixes it. That feedback edge is where the
accuracy comes from.

Repairs are bounded. Each one costs a model call, and a query that is still
failing on the third attempt is usually wrong in a way more attempts will not
resolve.
"""

from dataclasses import dataclass, field

from askdb.agent import checks
from askdb.agent.prompts import (
    draft_prompt,
    extract_sql,
    recheck_prompt,
    repair_prompt,
)
from askdb.agent.tools import QueryOutput, Toolbox
from askdb.llm.client import ModelClient
from askdb.llm.types import Usage
from askdb.retrieval import documents, lexical

DEFAULT_MAX_REPAIRS = 2
DEFAULT_SCHEMA_TABLES = 5


@dataclass(frozen=True)
class Step:
    """One action taken while answering, for cost attribution and tracing."""

    kind: str
    detail: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str | None = None
    cached: bool = False


@dataclass(frozen=True)
class AgentResult:
    question: str
    sql: str | None
    output: QueryOutput | None
    steps: tuple[Step, ...]
    tables_shown: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return self.output is not None and self.output.ok

    @property
    def usage(self) -> Usage:
        total = Usage()
        for step in self.steps:
            total = total + step.usage
        return total

    @property
    def model_calls(self) -> int:
        return sum(1 for step in self.steps if step.kind in {"draft", "repair", "recheck"})

    @property
    def repairs(self) -> int:
        return sum(1 for step in self.steps if step.kind == "repair")

    @property
    def rechecks(self) -> int:
        return sum(1 for step in self.steps if step.kind == "recheck")

    @property
    def concerns(self) -> tuple[str, ...]:
        return tuple(step.detail for step in self.steps if step.kind == "suspicion")


class Agent:
    def __init__(
        self,
        toolbox: Toolbox,
        client: ModelClient,
        *,
        model: str | None = None,
        max_repairs: int = DEFAULT_MAX_REPAIRS,
        schema_tables: int = DEFAULT_SCHEMA_TABLES,
        ground_values: bool = False,
        self_check: bool = True,
    ) -> None:
        self.toolbox = toolbox
        self.client = client
        self.model = model
        self.max_repairs = max_repairs
        self.schema_tables = schema_tables
        self.ground_values = ground_values
        self.self_check = self_check
        self._index = lexical.BM25(documents.describe_catalog(toolbox.catalog))

    def select_tables(self, question: str, evidence: str = "") -> tuple[str, ...]:
        """Which tables to put in the prompt.

        A schema small enough to fit goes in whole: retrieval can only lose
        information here, and most Mini-Dev databases have eight tables or
        fewer.
        """
        available = tuple(table.name for table in self.toolbox.catalog.tables)
        if len(available) <= self.schema_tables:
            return available

        query = f"{question} {evidence}".strip()
        ranked = self._index.search(query, limit=self.schema_tables)
        return tuple(result.table for result in ranked)

    def answer(self, question: str, evidence: str = "") -> AgentResult:
        tables = self.select_tables(question, evidence)
        schema = (
            self.toolbox.schema_context(tables)
            if self.ground_values
            else self.toolbox.catalog.to_ddl(tables=list(tables))
        )

        steps: list[Step] = []
        prompt = draft_prompt(schema, question, evidence)
        kind = "draft"
        sql: str | None = None
        output: QueryOutput | None = None

        # The best successful attempt seen so far. A recheck can make the
        # answer worse, so a clean result is never discarded for a suspicious
        # one.
        best_sql: str | None = None
        best_output: QueryOutput | None = None
        best_concern: checks.Suspicion | None = None
        rechecked = False

        for attempt in range(self.max_repairs + 1):
            completion = self.client.complete(prompt, model=self.model)
            sql = extract_sql(completion.text)
            steps.append(
                Step(
                    kind=kind,
                    detail=sql,
                    usage=completion.usage,
                    latency_ms=completion.latency_ms,
                    model=completion.model,
                    cached=completion.cached,
                )
            )

            output = self.toolbox.execute_sql(sql)
            steps.append(
                Step(
                    kind="execute",
                    detail=output.error or f"{len(output.rows)} rows",
                    latency_ms=output.duration_ms,
                )
            )

            if not output.ok:
                if attempt == self.max_repairs:
                    break
                kind = "repair"
                prompt = repair_prompt(
                    schema, question, sql, output.error or "unknown error", evidence
                )
                continue

            concern = checks.inspect(question, output) if self.self_check else None

            if best_output is None or (best_concern is not None and concern is None):
                best_sql, best_output, best_concern = sql, output, concern

            if concern is None or rechecked or attempt == self.max_repairs:
                break

            rechecked = True
            steps.append(Step(kind="suspicion", detail=concern.code))
            kind = "recheck"
            prompt = recheck_prompt(schema, question, sql, concern.message, evidence)

        return AgentResult(
            question=question,
            sql=best_sql if best_output is not None else sql,
            output=best_output if best_output is not None else output,
            steps=tuple(steps),
            tables_shown=tables,
        )
