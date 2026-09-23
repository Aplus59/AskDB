"""The HTTP service.

Built so the whole thing can be exercised without a model or a network: the
agent factory is injectable, so tests drive real requests through real routing
against a scripted model.
"""

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException

from askdb.agent.loop import Agent, AgentResult
from askdb.agent.tools import Toolbox
from askdb.api.registry import DatabaseRegistry, UnknownDatabase
from askdb.api.schemas import (
    AskRequest,
    AskResponse,
    DatabaseSummary,
    HealthResponse,
    Meta,
    TraceStep,
)
from askdb.config import settings
from askdb.data import minidev

AgentFactory = Callable[[Toolbox, int | None], Agent]


def _default_agent_factory(toolbox: Toolbox, samples: int | None) -> Agent:
    # Imported here so that constructing the app does not require an API key.
    from askdb.llm.factory import build_client

    return Agent(toolbox, build_client(), samples=samples or 1)


def to_response(request: AskRequest, result: AgentResult) -> AskResponse:
    output = result.output
    verdict = result.verdict

    return AskResponse(
        question=request.question,
        database=request.database,
        decision=verdict.decision if verdict else "answer",
        confidence=verdict.confidence if verdict else 1.0,
        reasons=list(verdict.reasons) if verdict else [],
        sql=result.sql,
        columns=list(output.columns) if output else [],
        # Rows are withheld unless the system is willing to stand behind them.
        # Returning a result alongside "I am not confident in this" invites the
        # reader to use it anyway.
        rows=[list(row) for row in output.rows] if output and result.answered else [],
        truncated=output.truncated if output else False,
        error=output.error if output else None,
        tables_considered=list(result.tables_shown),
        trace=[
            TraceStep(
                kind=step.kind,
                detail=step.detail,
                model=step.model,
                tokens=step.usage.total_tokens,
                latency_ms=step.latency_ms,
                cached=step.cached,
            )
            for step in result.steps
        ],
        meta=Meta(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            model_calls=result.model_calls,
            repairs=result.repairs,
            latency_ms=sum(step.latency_ms for step in result.steps),
            agreement=result.agreement,
        ),
    )


def create_app(
    databases_root: Path | None = None,
    agent_factory: AgentFactory = _default_agent_factory,
) -> FastAPI:
    if databases_root is None:
        try:
            databases_root = minidev.locate_databases(settings.data_dir / "mini_dev")
        except minidev.DatasetError:
            databases_root = settings.databases

    registry = DatabaseRegistry(databases_root)
    app = FastAPI(title="askdb", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            databases=len(registry.available()),
            model=settings.model_small,
        )

    @app.get("/databases", response_model=list[DatabaseSummary])
    def databases() -> list[DatabaseSummary]:
        summaries = []
        for name in registry.available():
            catalog = registry.toolbox(name).catalog
            summaries.append(
                DatabaseSummary(
                    name=name,
                    tables=len(catalog.tables),
                    rows=sum(table.row_count for table in catalog.tables),
                )
            )
        return summaries

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            toolbox = registry.toolbox(request.database)
        except UnknownDatabase:
            raise HTTPException(
                status_code=404,
                detail=f"unknown database {request.database!r}; "
                f"available: {', '.join(registry.available()) or 'none'}",
            ) from None

        agent = agent_factory(toolbox, request.samples)
        result = agent.answer(request.question, request.evidence)
        return to_response(request, result)

    return app
