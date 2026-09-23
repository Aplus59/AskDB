import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from askdb.agent.loop import Agent
from askdb.agent.tools import Toolbox
from askdb.api.app import create_app
from askdb.llm.types import Completion, Usage


class ScriptedModel:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        variant: int = 0,
    ) -> Completion:
        text = self.responses.pop(0) if self.responses else "SELECT name FROM artist"
        return Completion(text=text, model="scripted", usage=Usage(80, 12), latency_ms=3.0)


@pytest.fixture
def databases_root(sample_db: Path, tmp_path: Path) -> Path:
    """A directory laid out the way the registry expects."""
    root = tmp_path / "dev_databases"
    (root / "music").mkdir(parents=True)
    shutil.copy(sample_db, root / "music" / "music.sqlite")
    return root


def build_client(databases_root: Path, *responses: str, samples: int = 1) -> TestClient:
    model = ScriptedModel(*responses)

    def factory(toolbox: Toolbox, requested: int | None) -> Agent:
        return Agent(toolbox, model, samples=requested or samples)

    return TestClient(create_app(databases_root, factory))


def test_health_reports_available_databases(databases_root: Path) -> None:
    response = build_client(databases_root).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["databases"] == 1


def test_databases_lists_shape(databases_root: Path) -> None:
    body = build_client(databases_root).get("/databases").json()

    assert body[0]["name"] == "music"
    assert body[0]["tables"] == 3
    assert body[0]["rows"] == 8


def test_ask_returns_rows_and_sql(databases_root: Path) -> None:
    client = build_client(databases_root, "SELECT name FROM artist ORDER BY name")
    body = client.post(
        "/ask", json={"question": "which artists are there", "database": "music"}
    ).json()

    assert body["decision"] == "answer"
    assert body["sql"] == "SELECT name FROM artist ORDER BY name"
    assert body["columns"] == ["name"]
    assert ["Miles Davis"] in body["rows"]


def test_ask_reports_cost_and_latency(databases_root: Path) -> None:
    client = build_client(databases_root, "SELECT name FROM artist")
    meta = client.post(
        "/ask", json={"question": "which artists", "database": "music"}
    ).json()["meta"]

    assert meta["input_tokens"] == 80
    assert meta["output_tokens"] == 12
    assert meta["model_calls"] == 1


def test_ask_includes_the_trace(databases_root: Path) -> None:
    client = build_client(databases_root, "SELECT name FROM artist")
    body = client.post(
        "/ask", json={"question": "which artists", "database": "music"}
    ).json()

    kinds = [step["kind"] for step in body["trace"]]
    assert kinds == ["draft", "execute"]


def test_ask_reports_which_tables_were_shown(databases_root: Path) -> None:
    client = build_client(databases_root, "SELECT name FROM artist")
    body = client.post(
        "/ask", json={"question": "which artists", "database": "music"}
    ).json()

    assert set(body["tables_considered"]) == {"artist", "album", "track"}


def test_rows_are_withheld_when_the_system_will_not_stand_behind_them(
    databases_root: Path,
) -> None:
    # Returning data alongside "I am not confident in this" invites the reader
    # to use it anyway.
    client = build_client(
        databases_root,
        "SELECT name FROM artist WHERE country = 'XX'",
        "SELECT name FROM artist WHERE country = 'YY'",
    )
    body = client.post(
        "/ask",
        json={"question": "what was the rainfall during the storm", "database": "music"},
    ).json()

    assert body["decision"] != "answer"
    assert body["rows"] == []
    assert body["reasons"]


def test_a_failing_query_surfaces_its_error(databases_root: Path) -> None:
    client = build_client(databases_root, *["SELECT nope FROM artist"] * 3)
    body = client.post(
        "/ask", json={"question": "which artists", "database": "music"}
    ).json()

    assert body["error"] is not None
    assert "nope" in body["error"]
    assert body["decision"] == "refuse"


def test_unknown_database_is_a_404_that_names_the_alternatives(
    databases_root: Path,
) -> None:
    client = build_client(databases_root)
    response = client.post(
        "/ask", json={"question": "anything", "database": "absent"}
    )

    assert response.status_code == 404
    assert "music" in response.json()["detail"]


def test_an_empty_question_is_rejected(databases_root: Path) -> None:
    response = build_client(databases_root).post(
        "/ask", json={"question": "", "database": "music"}
    )
    assert response.status_code == 422


def test_samples_can_be_requested_per_call(databases_root: Path) -> None:
    client = build_client(databases_root, *["SELECT name FROM artist"] * 3)
    body = client.post(
        "/ask",
        json={"question": "which artists", "database": "music", "samples": 3},
    ).json()

    assert body["meta"]["model_calls"] == 3
    assert body["meta"]["agreement"] == 1.0


def test_an_absurd_sample_count_is_rejected(databases_root: Path) -> None:
    response = build_client(databases_root).post(
        "/ask",
        json={"question": "which artists", "database": "music", "samples": 99},
    )
    assert response.status_code == 422


def test_registry_ignores_archive_metadata_directories(databases_root: Path) -> None:
    junk = databases_root / "__MACOSX"
    junk.mkdir()
    assert build_client(databases_root).get("/health").json()["databases"] == 1


def test_registry_skips_folders_without_a_database(databases_root: Path) -> None:
    (databases_root / "empty").mkdir()
    assert build_client(databases_root).get("/health").json()["databases"] == 1


def test_a_database_is_only_loaded_once(databases_root: Path) -> None:
    # Counting rows in every table is slow; doing it per request would make
    # the first question of every conversation needlessly expensive.
    from askdb.api.registry import DatabaseRegistry

    registry = DatabaseRegistry(databases_root)
    assert registry.toolbox("music") is registry.toolbox("music")


def test_writes_are_rejected_by_the_connection(databases_root: Path) -> None:
    client = build_client(databases_root, *["DELETE FROM artist"] * 3)
    body = client.post(
        "/ask", json={"question": "remove the artists", "database": "music"}
    ).json()

    assert body["error"] is not None

    # And the data is genuinely untouched.
    conn = sqlite3.connect(databases_root / "music" / "music.sqlite")
    assert conn.execute("SELECT COUNT(*) FROM artist").fetchone()[0] == 3
    conn.close()


def test_the_service_starts_with_no_databases_at_all(tmp_path: Path) -> None:
    # A fresh checkout has no dataset. Refusing to start would make the
    # service impossible to smoke-test before downloading 600 MB.
    client = build_client(tmp_path / "absent")

    assert client.get("/health").json()["databases"] == 0
    assert client.get("/databases").json() == []


def test_asking_with_no_databases_explains_rather_than_crashes(tmp_path: Path) -> None:
    client = build_client(tmp_path / "absent")
    response = client.post("/ask", json={"question": "anything", "database": "music"})

    assert response.status_code == 404
    assert "none" in response.json()["detail"]


def test_the_index_page_is_served(databases_root: Path) -> None:
    response = build_client(databases_root).get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "askdb" in response.text


def test_the_index_answers_head_requests(databases_root: Path) -> None:
    # Uptime checks and load balancers probe with HEAD, and FastAPI does not
    # add it implicitly alongside GET.
    assert build_client(databases_root).head("/").status_code == 200


def test_the_cache_can_live_apart_from_the_databases(tmp_path: Path) -> None:
    # In a container the databases are mounted read-only, so the response
    # cache cannot default to sitting beside them.
    from askdb.config import Settings

    settings = Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    assert settings.response_cache == tmp_path / "cache" / "responses.sqlite"


def test_the_cache_defaults_beside_the_data(tmp_path: Path) -> None:
    from askdb.config import Settings

    settings = Settings(data_dir=tmp_path / "data")
    assert settings.response_cache == tmp_path / "data" / "responses.sqlite"
