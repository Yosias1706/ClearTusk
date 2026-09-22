"""Offline evaluation over the annotated corpus.

A benchmark sweep re-cleans every corpus clip with its annotated call profile
and records per-clip metrics plus aggregate statistics. Those aggregates are
the headline numbers the analytics dashboard reports, and re-running the sweep
after a change to the DSP shows immediately whether quality moved.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from cleartusk import __version__
from cleartusk.audio.denoise import clean_signal
from cleartusk.audio.io import load_analysis_audio, resample, write_audio
from cleartusk.audio.presets import get_profile
from cleartusk.config import Settings, get_settings
from cleartusk.db.models import BenchmarkResult, BenchmarkRun
from cleartusk.services.corpus import load_annotations, noise_reference_path

logger = logging.getLogger(__name__)

METRIC_FIELDS = (
    "target_retention_pct",
    "machine_suppression_pct",
    "target_machine_gain_db",
    "spectral_fidelity",
    "core_rumble_retention_pct",
)


@dataclass(frozen=True, slots=True)
class ClipOutcome:
    """Result of cleaning one corpus clip."""

    clip_name: str
    call_type: str
    selected_preset: str
    candidate_score: float
    audio_seconds: float
    duration_ms: float
    used_noise_reference: bool
    metrics: dict[str, float | bool]


def _clip_jobs(settings: Settings, limit: int | None) -> list[tuple[str, str]]:
    """Pair each prepared clip with its annotated call type."""
    call_types = {annotation.clip_name: annotation.call_type for annotation in load_annotations(settings)}
    jobs = [
        (path.stem, call_types.get(path.stem, "default"))
        for path in sorted(settings.call_clips_dir.glob("*.wav"), key=lambda item: item.stem)
    ]
    return jobs[:limit] if limit else jobs


def process_clip(
    clip_name: str,
    call_type: str,
    use_noise_reference: bool = True,
    export_audio: bool = False,
) -> ClipOutcome:
    """Clean one corpus clip; safe to call inside a worker process."""
    settings = get_settings()
    clip_path = settings.call_clips_dir / f"{clip_name}.wav"
    profile = get_profile(call_type)

    audio = load_analysis_audio(clip_path, settings.sample_rate)
    noise_path = noise_reference_path(clip_name, settings) if use_noise_reference else None
    noise = load_analysis_audio(noise_path, settings.sample_rate) if noise_path else None

    started = time.perf_counter()
    result = clean_signal(audio, profile, noise_reference=noise, sample_rate=settings.sample_rate)
    elapsed_ms = (time.perf_counter() - started) * 1000

    if export_audio:
        destination = settings.processed_dir / f"{clip_name}.wav"
        write_audio(
            destination,
            resample(result.audio, settings.sample_rate, settings.playback_sample_rate),
            settings.playback_sample_rate,
        )

    return ClipOutcome(
        clip_name=clip_name,
        call_type=call_type,
        selected_preset=result.preset.name,
        candidate_score=result.score,
        audio_seconds=float(audio.size / settings.sample_rate),
        duration_ms=elapsed_ms,
        used_noise_reference=noise is not None,
        metrics=result.metrics.as_dict(),
    )


def _worker(payload: tuple[str, str, bool, bool]) -> ClipOutcome:
    return process_clip(*payload)


def run_benchmark(
    session: Session,
    label: str = "corpus sweep",
    limit: int | None = None,
    workers: int | None = None,
    use_noise_reference: bool = True,
    export_audio: bool = False,
    settings: Settings | None = None,
    progress: bool = False,
) -> BenchmarkRun:
    """Sweep the corpus and persist a :class:`BenchmarkRun` with its per-clip rows."""
    settings = settings or get_settings()
    jobs = _clip_jobs(settings, limit)
    if not jobs:
        raise RuntimeError(
            "No corpus clips found. Run `cleartusk prepare-corpus` to cut clips from the raw audio."
        )

    workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    payloads = [(name, call_type, use_noise_reference, export_audio) for name, call_type in jobs]

    started = time.perf_counter()
    outcomes: list[ClipOutcome] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, outcome in enumerate(pool.map(_worker, payloads, chunksize=4), start=1):
                outcomes.append(outcome)
                if progress and index % 25 == 0:
                    logger.info("benchmarked %d/%d clips", index, len(payloads))
    else:
        for index, payload in enumerate(payloads, start=1):
            outcomes.append(_worker(payload))
            if progress and index % 25 == 0:
                logger.info("benchmarked %d/%d clips", index, len(payloads))
    wall_seconds = time.perf_counter() - started

    total_audio = sum(outcome.audio_seconds for outcome in outcomes)
    benchmark = BenchmarkRun(
        label=label,
        pipeline_version=__version__,
        clip_count=len(outcomes),
        total_audio_seconds=total_audio,
        wall_seconds=wall_seconds,
        realtime_factor=total_audio / wall_seconds if wall_seconds > 0 else 0.0,
        aggregates=summarize(outcomes, wall_seconds, workers),
    )
    benchmark.results = [
        BenchmarkResult(
            clip_name=outcome.clip_name,
            call_type=outcome.call_type,
            selected_preset=outcome.selected_preset,
            candidate_score=outcome.candidate_score,
            audio_seconds=outcome.audio_seconds,
            duration_ms=outcome.duration_ms,
            used_noise_reference=outcome.used_noise_reference,
            target_retention_pct=float(outcome.metrics["target_retention_pct"]),
            machine_suppression_pct=float(outcome.metrics["machine_suppression_pct"]),
            target_machine_gain_db=float(outcome.metrics["target_machine_gain_db"]),
            spectral_fidelity=float(outcome.metrics["spectral_fidelity"]),
            core_rumble_retention_pct=float(outcome.metrics["core_rumble_retention_pct"]),
            peak_in_target_band=bool(outcome.metrics["peak_in_target_band"]),
        )
        for outcome in outcomes
    ]
    session.add(benchmark)
    session.flush()
    return benchmark


def _describe(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(median(values)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize(outcomes: Iterable[ClipOutcome], wall_seconds: float, workers: int) -> dict[str, Any]:
    """Aggregate per-clip outcomes into the dashboard payload."""
    outcomes = list(outcomes)
    if not outcomes:
        return {}

    by_type: dict[str, list[ClipOutcome]] = {}
    preset_counts: dict[str, int] = {}
    for outcome in outcomes:
        by_type.setdefault(outcome.call_type, []).append(outcome)
        preset_counts[outcome.selected_preset] = preset_counts.get(outcome.selected_preset, 0) + 1

    def metric_values(items: Sequence[ClipOutcome], field: str) -> list[float]:
        return [float(item.metrics[field]) for item in items]

    total_audio = sum(outcome.audio_seconds for outcome in outcomes)
    return {
        "clip_count": len(outcomes),
        "total_audio_seconds": total_audio,
        "wall_seconds": wall_seconds,
        "workers": workers,
        "realtime_factor": total_audio / wall_seconds if wall_seconds else 0.0,
        "clips_per_minute": len(outcomes) / (wall_seconds / 60) if wall_seconds else 0.0,
        "latency_ms": _describe([outcome.duration_ms for outcome in outcomes]),
        "peak_in_target_band_pct": 100
        * sum(1 for outcome in outcomes if outcome.metrics["peak_in_target_band"])
        / len(outcomes),
        "noise_reference_coverage_pct": 100
        * sum(1 for outcome in outcomes if outcome.used_noise_reference)
        / len(outcomes),
        "metrics": {field: _describe(metric_values(outcomes, field)) for field in METRIC_FIELDS},
        "by_call_type": {
            call_type: {
                "clip_count": len(items),
                **{field: _describe(metric_values(items, field))["mean"] for field in METRIC_FIELDS},
            }
            for call_type, items in sorted(by_type.items())
        },
        "preset_distribution": dict(sorted(preset_counts.items(), key=lambda item: -item[1])),
    }


def export_report(benchmark: BenchmarkRun, settings: Settings | None = None) -> Path:
    """Write a CSV of per-clip results next to the other runtime reports."""
    import csv

    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    path = settings.reports_dir / f"benchmark_{benchmark.public_id[:8]}.csv"
    fields = [
        "clip_name",
        "call_type",
        "selected_preset",
        "candidate_score",
        "audio_seconds",
        "duration_ms",
        "used_noise_reference",
        "target_retention_pct",
        "machine_suppression_pct",
        "target_machine_gain_db",
        "spectral_fidelity",
        "core_rumble_retention_pct",
        "peak_in_target_band",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in benchmark.results:
            writer.writerow({field: getattr(result, field) for field in fields})
    return path
