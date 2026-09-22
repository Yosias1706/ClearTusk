"""Application configuration.

Every filesystem location and service endpoint used by ClearTusk is resolved
here from environment variables (optionally via a local ``.env`` file), so no
module ever hard-codes a machine-specific path.
"""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards from ``start`` until a project marker is found."""
    current = (start or PACKAGE_DIR).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return PACKAGE_DIR.parent


class Settings(BaseSettings):
    """Runtime settings, all overridable with ``CLEARTUSK_*`` env variables."""

    model_config = SettingsConfigDict(
        env_prefix="CLEARTUSK_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Core ---------------------------------------------------------------
    environment: str = Field(default="development")
    secret_key: str = Field(default="dev-secret-change-me")
    log_level: str = Field(default="INFO")

    # --- Locations ----------------------------------------------------------
    project_root: Path = Field(default_factory=find_project_root)
    data_dir: Path | None = Field(default=None, description="Root of all datasets and artifacts.")
    corpus_dir: Path | None = Field(default=None)
    runtime_dir: Path | None = Field(default=None, description="Writable dir for uploads/models/db.")

    # --- Database -----------------------------------------------------------
    database_url: str | None = Field(
        default=None,
        description="SQLAlchemy URL. Defaults to a local SQLite file so the app runs unconfigured.",
    )
    database_echo: bool = Field(default=False)
    database_pool_size: int = Field(default=5)

    # --- Uploads ------------------------------------------------------------
    max_upload_mb: int = Field(default=64, ge=1, le=1024)
    allowed_extensions: tuple[str, ...] = Field(default=(".wav", ".mp3", ".flac", ".ogg", ".m4a"))

    # --- Signal processing --------------------------------------------------
    sample_rate: int = Field(default=2000, description="Analysis rate; elephant energy lives <1 kHz.")
    playback_sample_rate: int = Field(default=16000)
    analysis_fmin_hz: float = Field(default=1.0)
    analysis_fmax_hz: float = Field(default=1000.0)

    # --- Detector -----------------------------------------------------------
    detector_window_seconds: float = Field(default=1.5, gt=0)
    detector_hop_seconds: float = Field(default=0.25, gt=0)
    detector_threshold: float = Field(default=0.6, ge=0, le=1)
    detector_min_event_seconds: float = Field(default=0.6, gt=0)

    # --- Spectrograms -------------------------------------------------------
    spectrogram_max_hz: float = Field(default=700.0)
    spectrogram_n_fft: int = Field(default=4096)
    spectrogram_dpi: int = Field(default=140)

    # --- Housekeeping -------------------------------------------------------
    retain_run_artifacts: int = Field(
        default=200, ge=0, description="Max number of upload/cleaned artifact pairs kept on disk."
    )

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _split_extensions(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(
                item if item.startswith(".") else f".{item}"
                for item in (part.strip().lower() for part in value.split(","))
                if item
            )
        return value

    # --- Derived paths ------------------------------------------------------
    @property
    def resolved_data_dir(self) -> Path:
        return (self.data_dir or self.project_root / "data").resolve()

    @property
    def resolved_corpus_dir(self) -> Path:
        return (self.corpus_dir or self.resolved_data_dir / "corpus").resolve()

    @property
    def resolved_runtime_dir(self) -> Path:
        return (self.runtime_dir or self.resolved_data_dir / "runtime").resolve()

    @property
    def raw_audio_dir(self) -> Path:
        return self.resolved_corpus_dir / "raw"

    @property
    def call_clips_dir(self) -> Path:
        return self.resolved_corpus_dir / "calls"

    @property
    def noise_clips_dir(self) -> Path:
        return self.resolved_corpus_dir / "noise"

    @property
    def context_clips_dir(self) -> Path:
        return self.resolved_corpus_dir / "context"

    @property
    def annotations_csv(self) -> Path:
        return self.resolved_corpus_dir / "annotations.csv"

    @property
    def processed_dir(self) -> Path:
        return self.resolved_data_dir / "processed" / "cleaned"

    @property
    def media_dir(self) -> Path:
        return self.resolved_runtime_dir / "media"

    @property
    def models_dir(self) -> Path:
        return self.resolved_runtime_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.resolved_runtime_dir / "reports"

    @property
    def detector_model_path(self) -> Path:
        return self.models_dir / "call_detector.joblib"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.resolved_runtime_dir / 'cleartusk.db'}"

    @property
    def database_backend(self) -> str:
        return self.resolved_database_url.split(":", 1)[0].split("+", 1)[0]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_runtime_dirs(self) -> None:
        for directory in (self.media_dir, self.models_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


def reload_settings() -> Settings:
    """Drop the cached settings (used by tests that patch the environment)."""
    get_settings.cache_clear()
    return get_settings()
