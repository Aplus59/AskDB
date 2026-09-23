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

from askdb.agent import checks, confidence
from askdb.agent.consistency import Draw, vote
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
DEFAULT_SAMPLE_TEMPERATURE = 0.7


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
    verdict: confidence.Verdict | None = None
    agreement: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.output is not None and self.output.ok

    @property
    def answered(self) -> bool:
        """Whether the answer is actually offered to the user.

        A query can execute perfectly and still not be presented, if the
        confidence assessment declined to stand behind it.
        """
        return self.succeeded and (self.verdict is None or self.verdict.answered)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for step in self.steps:
            total = total + step.usage
        return total

    @property
    def model_calls(self) -> int:
        return sum(
            1
            for step in self.steps
            if step.kind in {"draft", "repair", "recheck", "sample"}
        )

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
        abstain: bool = True,
        samples: int = 1,
        sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
        clarify_below: float = confidence.DEFAULT_CLARIFY_BELOW,
        refuse_below: float = confidence.DEFAULT_REFUSE_BELOW,
    ) -> None:
        self.toolbox = toolbox
        self.client = client
        self.model = model
        self.max_repairs = max_repairs
        self.schema_tables = schema_tables
        self.ground_values = ground_values
        self.self_check = self_check
        self.abstain = abstain
        self.samples = max(1, samples)
        self.sample_temperature = sample_temperature
        self.clarify_below = clarify_below
        self.refuse_below = refuse_below
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

    def _execute(self, sql: str, steps: list[Step]) -> QueryOutput:
        output = self.toolbox.execute_sql(sql)
        steps.append(
            Step(
                kind="execute",
                detail=output.error or f"{len(output.rows)} rows",
                latency_ms=output.duration_ms,
            )
        )
        return output

    def _single_attempt(
        self, prompt: str, kind: str, steps: list[Step]
    ) -> tuple[str, QueryOutput]:
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
        return sql, self._execute(sql, steps)

    def _voted_draft(
        self, prompt: str, steps: list[Step]
    ) -> tuple[str, QueryOutput, float]:
        """Draw the query several times and keep the most agreed-upon answer.

        Only the first draft is sampled. Repairs are not: a repair already has
        the database's own error to work from, which is far better evidence
        than another opinion, and sampling every stage would multiply the cost
        of the whole loop rather than just its first step.
        """
        draws: list[Draw] = []
        for index in range(self.samples):
            completion = self.client.complete(
                prompt,
                model=self.model,
                temperature=self.sample_temperature,
                variant=index,
            )
            sql = extract_sql(completion.text)
            steps.append(
                Step(
                    kind="sample",
                    detail=sql,
                    usage=completion.usage,
                    latency_ms=completion.latency_ms,
                    model=completion.model,
                    cached=completion.cached,
                )
            )
            draws.append(Draw(sql=sql, output=self.toolbox.execute_sql(sql)))

        consensus = vote(draws)
        steps.append(
            Step(
                kind="vote",
                detail=(
                    f"{consensus.agreement:.2f} agreement across {consensus.draws} draws, "
                    f"{consensus.distinct_answers} distinct answers"
                ),
            )
        )
        steps.append(
            Step(
                kind="execute",
                detail=consensus.output.error or f"{len(consensus.output.rows)} rows",
                latency_ms=consensus.output.duration_ms,
            )
        )
        return consensus.sql, consensus.output, consensus.agreement

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

        agreement: float | None = None

        for attempt in range(self.max_repairs + 1):
            if kind == "draft" and self.samples > 1:
                sql, output, agreement = self._voted_draft(prompt, steps)
            else:
                sql, output = self._single_attempt(prompt, kind, steps)

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

        final_sql = best_sql if best_output is not None else sql
        final_output = best_output if best_output is not None else output

        verdict = None
        if self.abstain:
            verdict = confidence.assess(
                question,
                schema_tokens=self._vocabulary(tables),
                repairs=sum(1 for step in steps if step.kind == "repair"),
                unresolved_concern=best_concern is not None,
                query_failed=final_output is None or not final_output.ok,
                agreement=agreement,
                clarify_below=self.clarify_below,
                refuse_below=self.refuse_below,
            )

        return AgentResult(
            question=question,
            sql=final_sql,
            output=final_output,
            steps=tuple(steps),
            tables_shown=tables,
            verdict=verdict,
            agreement=agreement,
        )

    def _vocabulary(self, tables: tuple[str, ...]) -> set[str]:
        wanted = {name.lower() for name in tables}
        columns = [
            column.name
            for table in self.toolbox.catalog.tables
            if table.name.lower() in wanted
            for column in table.columns
        ]
        return confidence.schema_vocabulary(list(tables), columns)
