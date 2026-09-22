"""End-to-end processing pipeline.

    upload -> probe -> detect calls -> resolve call type -> denoise
           -> render spectrograms -> score -> persist

Every stage is timed, and the resulting row in ``processing_runs`` is what the
analytics dashboard aggregates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from cleartusk.audio.denoise import CleaningResult, clean_signal
from cleartusk.audio.detection import CallDetector, DetectionSummary
from cleartusk.audio.features import infer_call_type
from cleartusk.audio.io import (
    file_digest,
    load_analysis_audio,
    load_native_audio,
    probe,
    resample,
    write_audio,
)
from cleartusk.audio.metrics import QualityMetrics
from cleartusk.audio.presets import CallProfile, get_profile
from cleartusk.audio.spectrogram import render_comparison, render_spectrogram
from cleartusk.config import Settings, get_settings
from cleartusk.db.models import CallDetection, ProcessingRun, Recording, RunMetric
from cleartusk.services.storage import CLEANED, SPECTROGRAMS, UPLOADS, MediaStorage, get_storage

logger = logging.getLogger(__name__)

AUTO = "auto"


class PipelineError(RuntimeError):
    """Raised for user-correctable problems (bad file, empty audio)."""


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    """Everything the API and UI need about one completed run."""

    run_public_id: str
    recording_public_id: str
    original_filename: str
    call_type: str
    call_type_source: str
    preset: str
    candidates_evaluated: int
    score: float
    metrics: QualityMetrics
    detection: DetectionSummary
    audio_seconds: float
    duration_ms: float
    realtime_factor: float
    original_key: str
    cleaned_key: str
    original_spectrogram_key: str
    cleaned_spectrogram_key: str
    comparison_spectrogram_key: str
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_public_id,
            "recording_id": self.recording_public_id,
            "filename": self.original_filename,
            "call_type": self.call_type,
            "call_type_source": self.call_type_source,
            "preset": self.preset,
            "candidates_evaluated": self.candidates_evaluated,
            "score": round(self.score, 3),
            "audio_seconds": round(self.audio_seconds, 3),
            "duration_ms": round(self.duration_ms, 1),
            "realtime_factor": round(self.realtime_factor, 2),
            "metrics": {
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in self.metrics.as_dict().items()
            },
            "detection": self.detection.as_dict(),
            "stage_timings_ms": {key: round(value, 1) for key, value in self.stage_timings_ms.items()},
        }


class _Stopwatch:
    """Collects per-stage wall times in milliseconds."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._start = time.perf_counter()

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings[stage] = (now - self._start) * 1000
        self._start = now

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings.values()))


class CleaningPipeline:
    """Coordinates audio processing and persistence for one session."""

    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        storage: MediaStorage | None = None,
        detector: CallDetector | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.storage = storage or get_storage(self.settings)
        self.detector = detector if detector is not None else CallDetector.load()

    # --- public API ---------------------------------------------------------
    def process_upload(self, file_storage, call_type: str = AUTO) -> PipelineOutcome:
        """Validate and process a Werkzeug upload."""
        filename = getattr(file_storage, "filename", "") or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in self.settings.allowed_extensions:
            raise PipelineError(
                f"Unsupported file type '{suffix or 'unknown'}'. "
                f"Allowed: {', '.join(self.settings.allowed_extensions)}."
            )
        key, path = self.storage.save_stream(file_storage, UPLOADS, filename)
        return self._process(path, key, filename, call_type)

    def process_file(self, source: Path, call_type: str = AUTO) -> PipelineOutcome:
        """Process a file already on disk (CLI / batch entry point)."""
        key, path = self.storage.import_file(source, UPLOADS)
        return self._process(path, key, source.name, call_type)

    # --- internals ----------------------------------------------------------
    def _process(
        self, upload_path: Path, upload_key: str, original_filename: str, requested_call_type: str
    ) -> PipelineOutcome:
        watch = _Stopwatch()
        info = probe(upload_path)
        if info.duration_seconds <= 0:
            raise PipelineError("The uploaded file contains no audio.")

        recording = Recording(
            origin="upload",
            original_filename=original_filename,
            storage_key=upload_key,
            content_hash=file_digest(upload_path),
            media_type=upload_path.suffix.lstrip(".").lower(),
            size_bytes=info.size_bytes,
            duration_seconds=info.duration_seconds,
            sample_rate=info.sample_rate,
            channels=info.channels,
        )
        self.session.add(recording)
        self.session.flush()
        watch.mark("ingest")

        analysis_audio = load_analysis_audio(upload_path, self.settings.sample_rate)
        if analysis_audio.size == 0:
            raise PipelineError("The uploaded file decoded to an empty signal.")
        watch.mark("decode")

        detection = self.detector.detect(analysis_audio, self.settings.sample_rate)
        watch.mark("detect")

        profile, call_type_source = self._resolve_profile(requested_call_type, analysis_audio, detection)

        cleaning = clean_signal(analysis_audio, profile, sample_rate=self.settings.sample_rate)
        watch.mark("denoise")

        cleaned_key, cleaned_path = self.storage.reserve(
            CLEANED, f"{Path(original_filename).stem}-cleaned.wav"
        )
        export_rate = info.sample_rate or self.settings.sample_rate
        write_audio(
            cleaned_path,
            resample(cleaning.audio, self.settings.sample_rate, export_rate),
            export_rate,
        )
        watch.mark("export")

        spectrogram_keys = self._render_spectrograms(
            upload_path, cleaning, detection, Path(original_filename).stem
        )
        watch.mark("render")

        audio_seconds = float(info.duration_seconds)
        total_ms = watch.total_ms
        run = ProcessingRun(
            recording_id=recording.id,
            status="succeeded",
            requested_call_type=requested_call_type,
            resolved_call_type=profile.name,
            call_type_source=call_type_source,
            selected_preset=cleaning.preset.name,
            preset_candidates=cleaning.candidates_evaluated,
            candidate_score=cleaning.score,
            audio_seconds=audio_seconds,
            duration_ms=total_ms,
            realtime_factor=audio_seconds / (total_ms / 1000) if total_ms > 0 else None,
            cleaned_key=cleaned_key,
            original_spectrogram_key=spectrogram_keys["original"],
            cleaned_spectrogram_key=spectrogram_keys["cleaned"],
            comparison_spectrogram_key=spectrogram_keys["comparison"],
            detection_event_count=len(detection.events),
            detection_peak_confidence=detection.peak_confidence,
            detector_model=detection.model_name if self.detector.is_available else None,
        )
        run.metrics = RunMetric(**cleaning.metrics.as_dict())
        run.detections = [
            CallDetection(
                start_seconds=event.start_seconds,
                end_seconds=event.end_seconds,
                confidence=event.confidence,
                peak_frequency_hz=event.peak_frequency_hz,
                predicted_call_type=event.predicted_call_type,
            )
            for event in detection.events
        ]
        self.session.add(run)
        self.session.flush()

        self._prune_artifacts()

        return PipelineOutcome(
            run_public_id=run.public_id,
            recording_public_id=recording.public_id,
            original_filename=original_filename,
            call_type=profile.name,
            call_type_source=call_type_source,
            preset=cleaning.preset.name,
            candidates_evaluated=cleaning.candidates_evaluated,
            score=cleaning.score,
            metrics=cleaning.metrics,
            detection=detection,
            audio_seconds=audio_seconds,
            duration_ms=total_ms,
            realtime_factor=audio_seconds / (total_ms / 1000) if total_ms > 0 else 0.0,
            original_key=upload_key,
            cleaned_key=cleaned_key,
            original_spectrogram_key=spectrogram_keys["original"],
            cleaned_spectrogram_key=spectrogram_keys["cleaned"],
            comparison_spectrogram_key=spectrogram_keys["comparison"],
            stage_timings_ms=watch.timings,
        )

    def _resolve_profile(
        self, requested: str, audio: np.ndarray, detection: DetectionSummary
    ) -> tuple[CallProfile, str]:
        """Honour an explicit choice; otherwise infer from detections, then spectrum."""
        requested = (requested or AUTO).strip().lower()
        if requested and requested != AUTO:
            return get_profile(requested), "user"

        dominant = detection.dominant_call_type()
        if dominant:
            return get_profile(dominant), "auto"

        inferred, _ = infer_call_type(audio, self.settings.sample_rate)
        return get_profile(inferred), "auto"

    def _render_spectrograms(
        self,
        upload_path: Path,
        cleaning: CleaningResult,
        detection: DetectionSummary,
        stem: str,
    ) -> dict[str, str]:
        native_audio, native_rate = load_native_audio(upload_path)
        events = [(event.start_seconds, event.end_seconds) for event in detection.events]

        original_key, original_path = self.storage.reserve(SPECTROGRAMS, f"{stem}-original.png")
        render_spectrogram(native_audio, native_rate, original_path, "Original", events)

        cleaned_key, cleaned_path = self.storage.reserve(SPECTROGRAMS, f"{stem}-cleaned.png")
        render_spectrogram(cleaning.audio, self.settings.sample_rate, cleaned_path, "Cleaned", events)

        comparison_key, comparison_path = self.storage.reserve(SPECTROGRAMS, f"{stem}-compare.png")
        render_comparison(
            resample(native_audio, native_rate, self.settings.sample_rate),
            cleaning.audio,
            self.settings.sample_rate,
            comparison_path,
            events,
        )
        return {"original": original_key, "cleaned": cleaned_key, "comparison": comparison_key}

    def _prune_artifacts(self) -> None:
        keep = self.settings.retain_run_artifacts
        if keep <= 0:
            return
        for category, multiplier in ((UPLOADS, 1), (CLEANED, 1), (SPECTROGRAMS, 3)):
            self.storage.prune(category, keep * multiplier)
