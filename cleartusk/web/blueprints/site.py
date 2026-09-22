"""HTML routes: landing page, studio, run history, analytics, contact, media."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    g,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from cleartusk.audio.presets import CALL_PROFILES
from cleartusk.db.models import ContactMessage, ProcessingRun
from cleartusk.services.analytics import AnalyticsService, summarize_run
from cleartusk.services.demo import build_showcase
from cleartusk.services.storage import StorageError

logger = logging.getLogger(__name__)

site_bp = Blueprint("site", __name__)

RUNS_PER_PAGE = 15
NOISE_SHOWCASE = (
    ("airplane.png", "Aircraft", "Rotor and turbine noise sweeping across the call band."),
    ("vehicle.png", "Vehicles", "Engine harmonics and tyre rumble from nearby tracks."),
    ("generator.png", "Generators", "Steady camp-generator tones that mask contact calls."),
)


def media_url(key: str | None) -> str | None:
    """Public URL for a storage key."""
    return url_for("site.media", key=key) if key else None


def _analytics() -> AnalyticsService:
    return AnalyticsService(g.db, current_app.config["SETTINGS"])


@site_bp.get("/")
def home():
    analytics = _analytics()
    showcase = build_showcase(current_app.config["SETTINGS"])
    return render_template(
        "home.html",
        overview=analytics.overview(),
        detector=analytics.detector_scorecard(),
        corpus=analytics.corpus_breakdown(),
        showcase=showcase,
        noise_showcase=NOISE_SHOWCASE,
    )


@site_bp.get("/studio")
def studio():
    profiles = [CALL_PROFILES[name] for name in ("rumble", "trumpet", "roar", "default")]
    return render_template(
        "studio.html",
        profiles=profiles,
        max_upload_mb=current_app.config["SETTINGS"].max_upload_mb,
        allowed_extensions=current_app.config["SETTINGS"].allowed_extensions,
    )


@site_bp.get("/runs")
def runs():
    page = max(1, request.args.get("page", type=int, default=1))
    call_type = request.args.get("call_type") or None

    query = (
        select(ProcessingRun)
        .options(selectinload(ProcessingRun.metrics), selectinload(ProcessingRun.recording))
        .order_by(ProcessingRun.created_at.desc(), ProcessingRun.id.desc())
    )
    if call_type in CALL_PROFILES:
        query = query.where(ProcessingRun.resolved_call_type == call_type)

    offset = (page - 1) * RUNS_PER_PAGE
    rows = g.db.scalars(query.offset(offset).limit(RUNS_PER_PAGE + 1)).all()
    has_next = len(rows) > RUNS_PER_PAGE

    return render_template(
        "runs.html",
        runs=[summarize_run(run) for run in rows[:RUNS_PER_PAGE]],
        page=page,
        has_next=has_next,
        call_type=call_type,
        call_types=list(CALL_PROFILES),
        totals=_analytics().overview(),
    )


@site_bp.get("/runs/<public_id>")
def run_detail(public_id: str):
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
        abort(404, description="That processing run no longer exists.")

    return render_template(
        "run_detail.html",
        run=run,
        summary=summarize_run(run),
        profile=CALL_PROFILES.get(run.resolved_call_type),
        detections=sorted(run.detections, key=lambda item: item.start_seconds),
    )


@site_bp.get("/analytics")
def analytics():
    return render_template("analytics.html", data=_analytics().dashboard())


@site_bp.route("/contact", methods=["GET", "POST"])
def contact():
    form = dict.fromkeys(("name", "email", "subject", "message"), "")
    status = None

    if request.method == "POST":
        form = {field: str(request.form.get(field, "")).strip() for field in form}
        missing = [field for field, value in form.items() if not value]
        if missing:
            status = {"kind": "error", "message": "Please complete every field before sending."}
        elif "@" not in form["email"]:
            status = {"kind": "error", "message": "That email address does not look valid."}
        else:
            g.db.add(ContactMessage(**form))
            g.db.flush()
            status = {
                "kind": "success",
                "message": "Thanks — your message is stored and we'll get back to you.",
            }
            form = dict.fromkeys(form, "")

    return render_template("contact.html", status=status, form=form)


@site_bp.get("/media/<path:key>")
def media(key: str):
    """Serve a generated artifact from the runtime media directory."""
    storage = current_app.extensions["storage"]
    try:
        path: Path = storage.path_for(key)
    except StorageError:
        abort(404)
    if not path.is_file():
        abort(404)
    return send_file(path, conditional=True)


@site_bp.get("/healthz")
def healthz():
    from cleartusk.db import database_is_reachable

    healthy = database_is_reachable()
    return (
        {
            "status": "ok" if healthy else "degraded",
            "database": current_app.config["SETTINGS"].database_backend,
            "detector": current_app.extensions["detector"].is_available,
            "version": current_app.config["VERSION"],
        },
        200 if healthy else 503,
    )


@site_bp.get("/dashboard")
def legacy_dashboard():
    """The prototype's upload page lived here."""
    return redirect(url_for("site.studio"), code=301)
