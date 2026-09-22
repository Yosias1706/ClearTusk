"""JSON API (``/api/v1``).

Same services as the HTML views, so the numbers can never drift apart.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, current_app, g, jsonify, request, url_for
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from cleartusk import __version__
from cleartusk.audio.presets import CALL_PROFILES
from cleartusk.db.models import ProcessingRun
from cleartusk.services.analytics import AnalyticsService, summarize_run
from cleartusk.services.pipeline import AUTO, CleaningPipeline, PipelineError

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

MAX_PAGE_SIZE = 100


def _media_url(key: str | None) -> str | None:
    return url_for("site.media", key=key) if key else None


def _analytics() -> AnalyticsService:
    return AnalyticsService(g.db, current_app.config["SETTINGS"])


@api_bp.get("/")
def index() -> Any:
    """Discovery document listing the available endpoints."""
    return jsonify(
        {
            "service": "cleartusk",
            "version": __version__,
            "endpoints": {
                "POST /api/v1/process": "Upload an audio file and run the cleaning pipeline.",
                "GET /api/v1/runs": "List processing runs (page, page_size, call_type).",
                "GET /api/v1/runs/<id>": "Fetch one run with metrics and detections.",
                "GET /api/v1/analytics": "Full analytics payload used by the dashboard.",
                "GET /api/v1/call-types": "Supported call profiles and their protected bands.",
                "GET /api/v1/benchmarks": "Benchmark history.",
            },
        }
    )


@api_bp.get("/call-types")
def call_types() -> Any:
    return jsonify(
        {
            "call_types": [
                {
                    "name": profile.name,
                    "display_name": profile.display_name,
                    "description": profile.description,
                    "target_band_hz": list(profile.target_band),
                    "machine_band_hz": list(profile.machine_band),
                    "presets": [preset.name for preset in profile.presets],
                }
                for profile in CALL_PROFILES.values()
            ]
        }
    )


@api_bp.post("/process")
def process() -> Any:
    """Clean one uploaded recording and return metrics, detections, and asset URLs."""
    upload = request.files.get("audio_file") or request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"success": False, "error": "Attach an audio file as 'audio_file'."}), 400

    call_type = str(request.form.get("call_type", AUTO)).strip().lower() or AUTO
    if call_type != AUTO and call_type not in CALL_PROFILES:
        return jsonify(
            {
                "success": False,
                "error": f"Unknown call type '{call_type}'. Use one of: auto, {', '.join(CALL_PROFILES)}.",
            }
        ), 400

    pipeline = CleaningPipeline(
        g.db,
        current_app.config["SETTINGS"],
        storage=current_app.extensions["storage"],
        detector=current_app.extensions["detector"],
    )
    try:
        outcome = pipeline.process_upload(upload, call_type)
    except PipelineError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - unexpected DSP failure
        logger.exception("processing failed")
        return jsonify({"success": False, "error": f"Processing failed: {exc}"}), 500

    payload = outcome.as_dict()
    payload.update(
        success=True,
        assets={
            "original_audio": _media_url(outcome.original_key),
            "cleaned_audio": _media_url(outcome.cleaned_key),
            "original_spectrogram": _media_url(outcome.original_spectrogram_key),
            "cleaned_spectrogram": _media_url(outcome.cleaned_spectrogram_key),
            "comparison_spectrogram": _media_url(outcome.comparison_spectrogram_key),
        },
        links={"self": url_for("site.run_detail", public_id=outcome.run_public_id)},
    )
    return jsonify(payload), 201


@api_bp.get("/runs")
def list_runs() -> Any:
    page = max(1, request.args.get("page", type=int, default=1))
    page_size = min(MAX_PAGE_SIZE, max(1, request.args.get("page_size", type=int, default=20)))
    call_type = request.args.get("call_type")

    query = (
        select(ProcessingRun)
        .options(selectinload(ProcessingRun.metrics), selectinload(ProcessingRun.recording))
        .order_by(ProcessingRun.created_at.desc(), ProcessingRun.id.desc())
    )
    if call_type:
        query = query.where(ProcessingRun.resolved_call_type == call_type)

    rows = g.db.scalars(query.offset((page - 1) * page_size).limit(page_size + 1)).all()
    return jsonify(
        {
            "page": page,
            "page_size": page_size,
            "has_next": len(rows) > page_size,
            "runs": [summarize_run(run) for run in rows[:page_size]],
        }
    )


@api_bp.get("/runs/<public_id>")
def get_run(public_id: str) -> Any:
    run = g.db.scalar(
        select(ProcessingRun)
        .options(
            selectinload(ProcessingRun.metrics),
            selectinload(ProcessingRun.recording),
            selectinload(ProcessingRun.detections),
        )
        .where(ProcessingRun.public_id == public_id)
    )
    if run is None:
        return jsonify({"success": False, "error": "No run with that id."}), 404

    payload = summarize_run(run)
    payload["detections"] = [
        {
            "start_seconds": detection.start_seconds,
            "end_seconds": detection.end_seconds,
            "confidence": detection.confidence,
            "peak_frequency_hz": detection.peak_frequency_hz,
            "predicted_call_type": detection.predicted_call_type,
        }
        for detection in run.detections
    ]
    payload["assets"] = {
        "original_audio": _media_url(run.recording.storage_key if run.recording else None),
        "cleaned_audio": _media_url(run.cleaned_key),
        "original_spectrogram": _media_url(run.original_spectrogram_key),
        "cleaned_spectrogram": _media_url(run.cleaned_spectrogram_key),
        "comparison_spectrogram": _media_url(run.comparison_spectrogram_key),
    }
    return jsonify(payload)


@api_bp.get("/analytics")
def analytics() -> Any:
    return jsonify(_analytics().dashboard())


@api_bp.get("/benchmarks")
def benchmarks() -> Any:
    service = _analytics()
    latest = service.latest_benchmark()
    return jsonify(
        {
            "history": service.benchmark_history(limit=request.args.get("limit", type=int, default=10)),
            "latest": {
                "id": latest.public_id,
                "label": latest.label,
                "created_at": latest.created_at.isoformat(),
                "clip_count": latest.clip_count,
                "total_audio_seconds": latest.total_audio_seconds,
                "wall_seconds": latest.wall_seconds,
                "realtime_factor": latest.realtime_factor,
                "aggregates": latest.aggregates,
            }
            if latest
            else None,
        }
    )


@api_bp.get("/detector")
def detector() -> Any:
    scorecard = _analytics().detector_scorecard()
    if scorecard is None:
        return jsonify({"available": False, "error": "No trained detector registered."}), 404
    return jsonify({"available": current_app.extensions["detector"].is_available, **scorecard})
