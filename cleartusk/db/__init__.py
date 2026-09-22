"""Database engine, session management, and schema helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from cleartusk.config import Settings, get_settings
from cleartusk.db.models import Base

logger = logging.getLogger(__name__)

__all__ = [
    "Base",
    "create_db_engine",
    "get_engine",
    "get_sessionmaker",
    "session_scope",
    "init_database",
    "reset_engine",
    "database_is_reachable",
]


def create_db_engine(settings: Settings | None = None) -> Engine:
    """Build an engine for the configured backend with sane pooling defaults."""
    settings = settings or get_settings()
    url = settings.resolved_database_url
    kwargs: dict[str, object] = {"echo": settings.database_echo, "future": True}

    if url.startswith("sqlite"):
        # SQLite needs the parent directory to exist and benefits from WAL.
        settings.resolved_runtime_dir.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_pool_size * 2,
            pool_pre_ping=True,
        )

    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver callback
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    engine = create_db_engine()
    logger.debug("database engine created for %s", engine.url.render_as_string(hide_password=True))
    return engine


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def reset_engine() -> None:
    """Dispose the cached engine (used by tests and CLI reconfiguration)."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on error."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database(engine: Engine | None = None, *, drop: bool = False) -> None:
    """Create (or recreate) all tables."""
    engine = engine or get_engine()
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def database_is_reachable(engine: Engine | None = None) -> bool:
    try:
        engine = engine or get_engine()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("database unreachable: %s", exc)
        return False
