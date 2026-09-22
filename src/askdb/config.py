from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ASKDB_", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"

    # Databases are large, so allow pointing at an external drive.
    databases_dir: Path | None = None

    # Number of distinct values fetched when profiling a column.
    sample_values_limit: int = 20

    @property
    def databases(self) -> Path:
        return self.databases_dir or (self.data_dir / "databases")


settings = Settings()
