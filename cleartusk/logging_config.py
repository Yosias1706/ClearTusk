"""Logging setup shared by the web app and the CLI."""

from __future__ import annotations

import logging
import sys

from cleartusk.config import Settings, get_settings

_CONFIGURED = False
FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(settings: Settings | None = None, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    settings = settings or get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    for noisy in ("matplotlib", "numba", "PIL", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
