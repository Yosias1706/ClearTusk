"""Flask application factory."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from cleartusk import __version__
from cleartusk.audio.detection import CallDetector
from cleartusk.config import Settings, get_settings
from cleartusk.db import get_sessionmaker, init_database
from cleartusk.logging_config import configure_logging
from cleartusk.services.storage import get_storage

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, *, auto_create_schema: bool = True) -> Flask:
    """Build a configured Flask app.

    The detector checkpoint is loaded once per process and shared by requests;
    a database session is created per request and closed on teardown.
    """
    settings = settings or get_settings()
    configure_logging(settings)
    settings.ensure_runtime_dirs()

    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=settings.secret_key,
        MAX_CONTENT_LENGTH=settings.max_upload_bytes,
        JSON_SORT_KEYS=False,
        SETTINGS=settings,
        VERSION=__version__,
    )

    if auto_create_schema:
        try:
            init_database()
        except Exception as exc:  # pragma: no cover - depends on the environment
            logger.error("could not initialise the database schema: %s", exc)

    app.extensions["storage"] = get_storage(settings)
    app.extensions["detector"] = CallDetector.load()
    if not app.extensions["detector"].is_available:
        logger.warning("call detector unavailable - run `cleartusk train-detector` to enable auto detection")

    _register_session_handling(app)
    _register_blueprints(app)
    _register_error_handlers(app)
    _register_template_helpers(app, settings)
    return app


def _register_session_handling(app: Flask) -> None:
    session_factory = get_sessionmaker()

    @app.before_request
    def _open_session() -> None:
        g.db = session_factory()

    @app.teardown_request
    def _close_session(exception: BaseException | None) -> None:
        session = g.pop("db", None)
        if session is None:
            return
        try:
            if exception is None:
                session.commit()
            else:
                session.rollback()
        finally:
            session.close()


def _register_blueprints(app: Flask) -> None:
    from cleartusk.web.blueprints.api import api_bp
    from cleartusk.web.blueprints.site import site_bp

    app.register_blueprint(site_bp)
    app.register_blueprint(api_bp, url_prefix="/api/v1")


def _wants_json() -> bool:
    return request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json"


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(error: RequestEntityTooLarge):
        settings: Settings = app.config["SETTINGS"]
        message = f"That file is larger than the {settings.max_upload_mb} MB upload limit."
        if _wants_json():
            return jsonify({"success": False, "error": message}), 413
        return render_template("error.html", code=413, message=message), 413

    @app.errorhandler(HTTPException)
    def _http_error(error: HTTPException):
        if _wants_json():
            return jsonify({"success": False, "error": error.description}), error.code or 500
        return (
            render_template("error.html", code=error.code, message=error.description),
            error.code or 500,
        )

    @app.errorhandler(Exception)
    def _unexpected(error: Exception):  # pragma: no cover - safety net
        logger.exception("unhandled error: %s", error)
        message = "Something went wrong while processing that request."
        if _wants_json():
            return jsonify({"success": False, "error": message}), 500
        return render_template("error.html", code=500, message=message), 500


def _register_template_helpers(app: Flask, settings: Settings) -> None:
    from cleartusk.web.blueprints.site import media_url

    @app.context_processor
    def _inject_globals() -> dict[str, Any]:
        return {
            "app_version": app.config["VERSION"],
            "media_url": media_url,
            "detector_available": app.extensions["detector"].is_available,
            "database_backend": settings.database_backend,
        }

    @app.template_filter("number")
    def _number(value: Any, digits: int = 0) -> str:
        try:
            return f"{float(value):,.{digits}f}"
        except (TypeError, ValueError):
            return "—"

    @app.template_filter("duration")
    def _duration(seconds: Any) -> str:
        try:
            total = float(seconds)
        except (TypeError, ValueError):
            return "—"
        minutes, remainder = divmod(int(total), 60)
        return f"{minutes}:{remainder:02d}" if minutes else f"{total:.1f}s"

    @app.template_filter("timeago")
    def _timeago(value: Any) -> str:
        if not value:
            return "—"
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - moment
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60} min ago"
        if seconds < 86400:
            return f"{seconds // 3600} h ago"
        return f"{seconds // 86400} d ago"
