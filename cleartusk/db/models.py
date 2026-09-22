"""SQLAlchemy ORM models.

The schema is portable across PostgreSQL (production / CI) and SQLite (zero
configuration local runs); only backend-neutral column types are used.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_public_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    """Declarative base with a shared JSON type map."""

    type_annotation_map = {dict[str, Any]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )


class Recording(Base, TimestampMixin):
    """An audio asset known to the system: a corpus clip or a user upload."""

    __tablename__ = "recordings"
    __table_args__ = (Index("ix_recordings_origin_created", "origin", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(32), default=new_public_id, unique=True, index=True)
    origin: Mapped[str] = mapped_column(String(24), nullable=False)  # upload | corpus
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    media_type: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    sample_rate: Mapped[int | None] = mapped_column(Integer)
    channels: Mapped[int | None] = mapped_column(Integer)

    runs: Mapped[list[ProcessingRun]] = relationship(back_populates="recording", cascade="all, delete-orphan")


class ProcessingRun(Base, TimestampMixin):
    """One execution of the cleaning pipeline over one recording."""

    __tablename__ = "processing_runs"
    __table_args__ = (Index("ix_runs_status_created", "status", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(32), default=new_public_id, unique=True, index=True)
    recording_id: Mapped[int] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"), index=True)

    status: Mapped[str] = mapped_column(String(16), default="succeeded", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    requested_call_type: Mapped[str] = mapped_column(String(24), nullable=False)
    resolved_call_type: Mapped[str] = mapped_column(String(24), nullable=False)
    call_type_source: Mapped[str] = mapped_column(String(16), default="user")  # user | auto
    selected_preset: Mapped[str | None] = mapped_column(String(48))
    preset_candidates: Mapped[int] = mapped_column(Integer, default=0)
    candidate_score: Mapped[float | None] = mapped_column(Float)

    audio_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    realtime_factor: Mapped[float | None] = mapped_column(Float)

    cleaned_key: Mapped[str | None] = mapped_column(String(512))
    original_spectrogram_key: Mapped[str | None] = mapped_column(String(512))
    cleaned_spectrogram_key: Mapped[str | None] = mapped_column(String(512))
    comparison_spectrogram_key: Mapped[str | None] = mapped_column(String(512))

    detection_event_count: Mapped[int] = mapped_column(Integer, default=0)
    detection_peak_confidence: Mapped[float | None] = mapped_column(Float)
    detector_model: Mapped[str | None] = mapped_column(String(64))

    recording: Mapped[Recording] = relationship(back_populates="runs")
    metrics: Mapped[RunMetric | None] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False
    )
    detections: Mapped[list[CallDetection]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="CallDetection.start_seconds"
    )


class RunMetric(Base):
    """Quality metrics for a single processing run (1:1 with the run)."""

    __tablename__ = "run_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("processing_runs.id", ondelete="CASCADE"), unique=True, index=True
    )

    target_retention_pct: Mapped[float] = mapped_column(Float)
    machine_suppression_pct: Mapped[float] = mapped_column(Float)
    target_machine_gain_db: Mapped[float] = mapped_column(Float)
    spectral_fidelity: Mapped[float] = mapped_column(Float)
    core_rumble_retention_pct: Mapped[float] = mapped_column(Float)
    broadband_reduction_pct: Mapped[float] = mapped_column(Float)
    spectral_sparsity_pct: Mapped[float] = mapped_column(Float)
    peak_frequency_hz: Mapped[float] = mapped_column(Float)
    peak_in_target_band: Mapped[bool] = mapped_column(Boolean, default=False)
    target_band_low_hz: Mapped[float] = mapped_column(Float)
    target_band_high_hz: Mapped[float] = mapped_column(Float)
    machine_band_low_hz: Mapped[float] = mapped_column(Float)
    machine_band_high_hz: Mapped[float] = mapped_column(Float)

    run: Mapped[ProcessingRun] = relationship(back_populates="metrics")


class CallDetection(Base):
    """A detected call event inside a processed recording."""

    __tablename__ = "call_detections"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("processing_runs.id", ondelete="CASCADE"), index=True)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    peak_frequency_hz: Mapped[float | None] = mapped_column(Float)
    predicted_call_type: Mapped[str | None] = mapped_column(String(24))

    run: Mapped[ProcessingRun] = relationship(back_populates="detections")


class CorpusClip(Base, TimestampMixin):
    """An annotated clip from the reference corpus (ingested from annotations.csv)."""

    __tablename__ = "corpus_clips"
    __table_args__ = (UniqueConstraint("selection", name="uq_corpus_clip_selection"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    selection: Mapped[int] = mapped_column(Integer, index=True)
    clip_name: Mapped[str] = mapped_column(String(128), index=True)
    source_file: Mapped[str] = mapped_column(String(255), index=True)
    noise_source: Mapped[str | None] = mapped_column(String(64))
    raw_call_type: Mapped[str] = mapped_column(String(64))
    call_type: Mapped[str] = mapped_column(String(24), index=True)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    duration_seconds: Mapped[float] = mapped_column(Float)
    has_noise_reference: Mapped[bool] = mapped_column(Boolean, default=False)
    has_context: Mapped[bool] = mapped_column(Boolean, default=False)


class BenchmarkRun(Base, TimestampMixin):
    """An offline evaluation sweep over the whole corpus."""

    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(32), default=new_public_id, unique=True, index=True)
    label: Mapped[str] = mapped_column(String(120))
    pipeline_version: Mapped[str] = mapped_column(String(32))
    clip_count: Mapped[int] = mapped_column(Integer, default=0)
    total_audio_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    wall_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    realtime_factor: Mapped[float] = mapped_column(Float, default=0.0)
    aggregates: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    results: Mapped[list[BenchmarkResult]] = relationship(
        back_populates="benchmark", cascade="all, delete-orphan"
    )


class BenchmarkResult(Base):
    """Per-clip outcome inside a benchmark run."""

    __tablename__ = "benchmark_results"
    __table_args__ = (Index("ix_benchmark_results_run_type", "benchmark_run_id", "call_type"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    benchmark_run_id: Mapped[int] = mapped_column(
        ForeignKey("benchmark_runs.id", ondelete="CASCADE"), index=True
    )
    clip_name: Mapped[str] = mapped_column(String(128))
    call_type: Mapped[str] = mapped_column(String(24))
    selected_preset: Mapped[str] = mapped_column(String(48))
    candidate_score: Mapped[float] = mapped_column(Float)
    audio_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    used_noise_reference: Mapped[bool] = mapped_column(Boolean, default=False)
    target_retention_pct: Mapped[float] = mapped_column(Float)
    machine_suppression_pct: Mapped[float] = mapped_column(Float)
    target_machine_gain_db: Mapped[float] = mapped_column(Float)
    spectral_fidelity: Mapped[float] = mapped_column(Float)
    core_rumble_retention_pct: Mapped[float] = mapped_column(Float)
    peak_in_target_band: Mapped[bool] = mapped_column(Boolean, default=False)

    benchmark: Mapped[BenchmarkRun] = relationship(back_populates="results")


class ModelVersion(Base, TimestampMixin):
    """A trained detector checkpoint plus its cross-validated scorecard."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    algorithm: Mapped[str] = mapped_column(String(64))
    artifact_key: Mapped[str] = mapped_column(String(512))
    feature_count: Mapped[int] = mapped_column(Integer, default=0)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    positive_count: Mapped[int] = mapped_column(Integer, default=0)
    negative_count: Mapped[int] = mapped_column(Integer, default=0)
    cv_folds: Mapped[int] = mapped_column(Integer, default=0)
    cv_strategy: Mapped[str] = mapped_column(String(120), default="")
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    precision: Mapped[float] = mapped_column(Float, default=0.0)
    recall: Mapped[float] = mapped_column(Float, default=0.0)
    f1: Mapped[float] = mapped_column(Float, default=0.0)
    roc_auc: Mapped[float] = mapped_column(Float, default=0.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ContactMessage(Base, TimestampMixin):
    """Inbound message from the contact form."""

    __tablename__ = "contact_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), index=True)
    subject: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    is_handled: Mapped[bool] = mapped_column(Boolean, default=False)
