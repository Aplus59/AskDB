from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ASKDB_", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"

    # Databases are large, so allow pointing at an external drive.
    databases_dir: Path | None = None

    # Number of distinct values fetched when profiling a column.
    sample_values_limit: int = 20

    # The two rungs of the cost ladder. Exact versions rather than the
    # "-latest" aliases: an alias can change under you between runs, which
    # would silently move a published baseline with no change to the code.
    #
    # Free-tier capacity fluctuates minute to minute, and the `-lite` variants
    # swing hardest: unavailable for stretches, then answering in under a
    # second while the full model takes twenty. Since FallbackClient degrades
    # past an unavailable model, lite belongs first — it is both the cheap rung
    # and the fast one whenever it has capacity.
    # Override with ASKDB_MODEL_SMALL / ASKDB_MODEL_LARGE in .env.
    model_small: str = "gemini-3.5-flash-lite"
    model_large: str = "gemini-3.6-flash"

    request_timeout_seconds: float = 60.0

    # Read from .env under its conventional name rather than the ASKDB_ prefix,
    # so the same variable works for any Google tooling in the project.
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")

    @property
    def databases(self) -> Path:
        return self.databases_dir or (self.data_dir / "databases")

    @property
    def response_cache(self) -> Path:
        return self.data_dir / "responses.sqlite"


settings = Settings()
