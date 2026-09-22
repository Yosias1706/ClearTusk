"""Analytics layer.

Every number shown on the dashboard is derived here from the database, so the
UI, the JSON API, and the CLI all report exactly the same statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session, selectinload

from cleartusk.audio.presets import CALL_PROFILES
from cleartusk.config import Settings, get_settings
from cleartusk.db.models import (
    BenchmarkRun,
    CallDetection,
    ContactMessage,
    CorpusClip,
    ModelVersion,
    ProcessingRun,
    Recording,
    RunMetric,
)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def _round(value: float | None, digits: int = 2) -> float:
    return round(float(value), digits) if value is not None else 0.0


@dataclass(frozen=True, slots=True)
class Histogram:
    """Bin counts ready to hand to a chart."""

    labels: list[str]
    counts: list[int]

    def as_dict(self) -> dict[str, list[Any]]:
        return {"labels": self.labels, "counts": self.counts}


def histogram(values: Sequence[float], bins: Sequence[float], unit: str = "") -> Histogram:
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=list(bins))
    labels = [f"{edges[index]:g}–{edges[index + 1]:g}{unit}" for index in range(len(edges) - 1)]
    return Histogram(labels=labels, counts=[int(count) for count in counts])


class AnalyticsService:
    """Read-only aggregations over runs, benchmarks, corpus, and models."""

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()

    # --- headline ----------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        runs_total = self.session.scalar(select(func.count(ProcessingRun.id))) or 0
        runs_ok = (
            self.session.scalar(
                select(func.count(ProcessingRun.id)).where(ProcessingRun.status == "succeeded")
            )
            or 0
        )
        audio_seconds = self.session.scalar(select(func.sum(ProcessingRun.audio_seconds))) or 0.0
        corpus_seconds = self.session.scalar(select(func.sum(CorpusClip.duration_seconds))) or 0.0
        benchmark = self.latest_benchmark()
        benchmark_metrics = (benchmark.aggregates or {}).get("metrics", {}) if benchmark else {}

        return {
            "runs_total": int(runs_total),
            "runs_succeeded": int(runs_ok),
            "success_rate_pct": _round(100 * runs_ok / runs_total, 1) if runs_total else 100.0,
            "audio_processed_minutes": _round(float(audio_seconds) / 60, 1),
            "detections_total": int(self.session.scalar(select(func.count(CallDetection.id))) or 0),
            "corpus_clips": int(self.session.scalar(select(func.count(CorpusClip.id))) or 0),
            "corpus_recordings": int(
                self.session.scalar(select(func.count(func.distinct(CorpusClip.source_file)))) or 0
            ),
            "corpus_minutes": _round(float(corpus_seconds) / 60, 1),
            "benchmark_clip_count": int(benchmark.clip_count) if benchmark else 0,
            "avg_target_retention_pct": _round(
                benchmark_metrics.get("target_retention_pct", {}).get("mean"), 1
            ),
            "avg_machine_suppression_pct": _round(
                benchmark_metrics.get("machine_suppression_pct", {}).get("mean"), 1
            ),
            "avg_gain_db": _round(benchmark_metrics.get("target_machine_gain_db", {}).get("mean"), 1),
            "avg_spectral_fidelity": _round(benchmark_metrics.get("spectral_fidelity", {}).get("mean"), 3),
            "benchmark_realtime_factor": _round(benchmark.realtime_factor, 1) if benchmark else 0.0,
            "benchmark_at": benchmark.created_at.isoformat() if benchmark else None,
        }

    # --- benchmarks --------------------------------------------------------
    def latest_benchmark(self) -> BenchmarkRun | None:
        return self.session.scalar(
            select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc(), BenchmarkRun.id.desc()).limit(1)
        )

    def benchmark_history(self, limit: int = 10) -> list[dict[str, Any]]:
        runs = self.session.scalars(
            select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).limit(limit)
        ).all()
        history = []
        for run in runs:
            metrics = (run.aggregates or {}).get("metrics", {})
            history.append(
                {
                    "id": run.public_id,
                    "label": run.label,
                    "created_at": run.created_at.isoformat(),
                    "clip_count": run.clip_count,
                    "realtime_factor": _round(run.realtime_factor, 1),
                    "retention": _round(metrics.get("target_retention_pct", {}).get("mean"), 1),
                    "suppression": _round(metrics.get("machine_suppression_pct", {}).get("mean"), 1),
                    "gain_db": _round(metrics.get("target_machine_gain_db", {}).get("mean"), 2),
                    "fidelity": _round(metrics.get("spectral_fidelity", {}).get("mean"), 3),
                }
            )
        return list(reversed(history))

    def quality_by_call_type(self) -> list[dict[str, Any]]:
        benchmark = self.latest_benchmark()
        rows = (benchmark.aggregates or {}).get("by_call_type", {}) if benchmark else {}
        return [
            {
                "call_type": call_type,
                "display_name": CALL_PROFILES[call_type].display_name
                if call_type in CALL_PROFILES
                else call_type.title(),
                "clip_count": int(values.get("clip_count", 0)),
                "retention": _round(values.get("target_retention_pct"), 1),
                "suppression": _round(values.get("machine_suppression_pct"), 1),
                "gain_db": _round(values.get("target_machine_gain_db"), 2),
                "fidelity": _round(values.get("spectral_fidelity"), 3),
            }
            for call_type, values in rows.items()
        ]

    def preset_distribution(self) -> dict[str, int]:
        benchmark = self.latest_benchmark()
        if benchmark and (benchmark.aggregates or {}).get("preset_distribution"):
            return dict(benchmark.aggregates["preset_distribution"])
        rows = self.session.execute(
            select(ProcessingRun.selected_preset, func.count(ProcessingRun.id))
            .where(ProcessingRun.selected_preset.is_not(None))
            .group_by(ProcessingRun.selected_preset)
            .order_by(func.count(ProcessingRun.id).desc())
        ).all()
        return {str(name): int(count) for name, count in rows}

    def quality_distributions(self) -> dict[str, dict[str, list[Any]]]:
        benchmark = self.latest_benchmark()
        if not benchmark:
            return {}
        retention = [result.target_retention_pct for result in benchmark.results]
        gain = [result.target_machine_gain_db for result in benchmark.results]
        return {
            "retention": histogram(retention, [0, 20, 40, 60, 80, 90, 100], "%").as_dict(),
            "gain_db": histogram(gain, [0, 5, 10, 15, 20, 25, 40], " dB").as_dict(),
        }

    # --- live pipeline -----------------------------------------------------
    def performance(self) -> dict[str, Any]:
        durations = list(
            self.session.scalars(select(ProcessingRun.duration_ms).where(ProcessingRun.duration_ms > 0)).all()
        )
        factors = list(
            self.session.scalars(
                select(ProcessingRun.realtime_factor).where(ProcessingRun.realtime_factor.is_not(None))
            ).all()
        )
        benchmark = self.latest_benchmark()
        aggregates = benchmark.aggregates if benchmark else {}
        return {
            "run_count": len(durations),
            "latency_p50_ms": _round(percentile(durations, 50), 0),
            "latency_p95_ms": _round(percentile(durations, 95), 0),
            "latency_max_ms": _round(max(durations) if durations else 0, 0),
            "realtime_factor_mean": _round(float(np.mean(factors)) if factors else 0.0, 1),
            "batch_realtime_factor": _round(aggregates.get("realtime_factor"), 1),
            "batch_clips_per_minute": _round(aggregates.get("clips_per_minute"), 0),
            "batch_workers": int(aggregates.get("workers", 0) or 0),
        }

    def throughput_timeline(self, days: int = 14) -> dict[str, list[Any]]:
        """Runs per day over the trailing window (zero-filled)."""
        since = datetime.now(UTC) - timedelta(days=days - 1)
        rows = self.session.execute(
            select(func.date(ProcessingRun.created_at), func.count(ProcessingRun.id))
            .where(ProcessingRun.created_at >= since)
            .group_by(func.date(ProcessingRun.created_at))
        ).all()
        counts = {str(day): int(count) for day, count in rows}
        labels = [(since + timedelta(days=offset)).date().isoformat() for offset in range(days)]
        return {"labels": labels, "counts": [counts.get(label, 0) for label in labels]}

    def call_type_usage(self) -> dict[str, int]:
        rows = self.session.execute(
            select(ProcessingRun.resolved_call_type, func.count(ProcessingRun.id)).group_by(
                ProcessingRun.resolved_call_type
            )
        ).all()
        return {str(call_type): int(count) for call_type, count in rows}

    def live_metric_means(self) -> dict[str, float]:
        row = self.session.execute(
            select(
                func.avg(RunMetric.target_retention_pct),
                func.avg(RunMetric.machine_suppression_pct),
                func.avg(RunMetric.target_machine_gain_db),
                func.avg(RunMetric.spectral_fidelity),
            )
        ).one()
        return {
            "retention": _round(row[0], 1),
            "suppression": _round(row[1], 1),
            "gain_db": _round(row[2], 2),
            "fidelity": _round(row[3], 3),
        }

    # --- corpus & models ---------------------------------------------------
    def corpus_breakdown(self) -> dict[str, Any]:
        by_noise = self.session.execute(
            select(CorpusClip.noise_source, func.count(CorpusClip.id), func.sum(CorpusClip.duration_seconds))
            .group_by(CorpusClip.noise_source)
            .order_by(func.count(CorpusClip.id).desc())
        ).all()
        by_type = self.session.execute(
            select(CorpusClip.call_type, func.count(CorpusClip.id)).group_by(CorpusClip.call_type)
        ).all()
        coverage = self.session.execute(
            select(
                func.count(CorpusClip.id),
                func.sum(func.cast(CorpusClip.has_noise_reference, Integer)),
                func.sum(func.cast(CorpusClip.has_context, Integer)),
            )
        ).one()
        total = int(coverage[0] or 0)
        return {
            "by_noise_source": [
                {
                    "noise_source": str(source or "unknown"),
                    "clips": int(count),
                    "minutes": _round(float(seconds or 0) / 60, 1),
                }
                for source, count, seconds in by_noise
            ],
            "by_call_type": {str(call_type): int(count) for call_type, count in by_type},
            "clips": total,
            "noise_reference_pct": _round(100 * int(coverage[1] or 0) / total, 1) if total else 0.0,
            "context_pct": _round(100 * int(coverage[2] or 0) / total, 1) if total else 0.0,
        }

    def detector_scorecard(self) -> dict[str, Any] | None:
        model = self.session.scalar(
            select(ModelVersion)
            .where(ModelVersion.is_active.is_(True))
            .order_by(ModelVersion.created_at.desc())
            .limit(1)
        )
        if model is None:
            return None
        return {
            "name": model.name,
            "algorithm": model.algorithm,
            "trained_at": model.created_at.isoformat(),
            "sample_count": model.sample_count,
            "positive_count": model.positive_count,
            "negative_count": model.negative_count,
            "feature_count": model.feature_count,
            "cv_folds": model.cv_folds,
            "cv_strategy": model.cv_strategy,
            "accuracy": _round(model.accuracy, 4),
            "precision": _round(model.precision, 4),
            "recall": _round(model.recall, 4),
            "f1": _round(model.f1, 4),
            "roc_auc": _round(model.roc_auc, 4),
            "top_features": (model.details or {}).get("top_features", [])[:6],
            "fold_scores": (model.details or {}).get("fold_scores", []),
        }

    # --- run feed ----------------------------------------------------------
    def recent_runs(self, limit: int = 12) -> list[dict[str, Any]]:
        runs = self.session.scalars(
            select(ProcessingRun)
            .options(selectinload(ProcessingRun.metrics), selectinload(ProcessingRun.recording))
            .order_by(ProcessingRun.created_at.desc(), ProcessingRun.id.desc())
            .limit(limit)
        ).all()
        return [summarize_run(run) for run in runs]

    def run_count(self) -> int:
        return int(self.session.scalar(select(func.count(ProcessingRun.id))) or 0)

    def unread_messages(self) -> int:
        return int(
            self.session.scalar(
                select(func.count(ContactMessage.id)).where(ContactMessage.is_handled.is_(False))
            )
            or 0
        )

    def storage_summary(self) -> dict[str, Any]:
        recordings = self.session.scalar(select(func.count(Recording.id))) or 0
        stored_bytes = self.session.scalar(select(func.sum(Recording.size_bytes))) or 0
        return {
            "recordings": int(recordings),
            "stored_megabytes": _round(float(stored_bytes) / (1024 * 1024), 1),
            "database_backend": self.settings.database_backend,
        }

    # --- composite ---------------------------------------------------------
    def dashboard(self) -> dict[str, Any]:
        return {
            "overview": self.overview(),
            "performance": self.performance(),
            "live_metrics": self.live_metric_means(),
            "quality_by_call_type": self.quality_by_call_type(),
            "quality_distributions": self.quality_distributions(),
            "preset_distribution": self.preset_distribution(),
            "call_type_usage": self.call_type_usage(),
            "throughput": self.throughput_timeline(),
            "benchmark_history": self.benchmark_history(),
            "corpus": self.corpus_breakdown(),
            "detector": self.detector_scorecard(),
            "storage": self.storage_summary(),
            "recent_runs": self.recent_runs(8),
        }


def summarize_run(run: ProcessingRun) -> dict[str, Any]:
    """Flatten a run (and its 1:1 metrics) for tables and JSON responses."""
    metrics = run.metrics
    return {
        "id": run.public_id,
        "created_at": run.created_at.isoformat(),
        "filename": run.recording.original_filename if run.recording else "",
        "status": run.status,
        "call_type": run.resolved_call_type,
        "call_type_source": run.call_type_source,
        "preset": run.selected_preset,
        "audio_seconds": _round(run.audio_seconds, 2),
        "duration_ms": _round(run.duration_ms, 0),
        "realtime_factor": _round(run.realtime_factor, 1),
        "events": run.detection_event_count,
        "peak_confidence": _round(run.detection_peak_confidence, 3),
        "retention": _round(metrics.target_retention_pct, 1) if metrics else None,
        "suppression": _round(metrics.machine_suppression_pct, 1) if metrics else None,
        "gain_db": _round(metrics.target_machine_gain_db, 2) if metrics else None,
        "fidelity": _round(metrics.spectral_fidelity, 3) if metrics else None,
    }
