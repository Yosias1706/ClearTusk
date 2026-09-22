"""Media storage.

Artifacts never live in the source tree: everything the app generates is keyed
into a configurable runtime directory (``CLEARTUSK_RUNTIME_DIR``) and served
through a single guarded route. Swapping in object storage later only requires
another implementation of :class:`MediaStorage`.
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from cleartusk.config import Settings, get_settings

logger = logging.getLogger(__name__)

UPLOADS = "uploads"
CLEANED = "cleaned"
SPECTROGRAMS = "spectrograms"


class StorageError(RuntimeError):
    """Raised when a key escapes the storage root or cannot be written."""


class MediaStorage:
    """Content-addressed-ish local filesystem storage with date-sharded keys."""

    def __init__(self, root: Path | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.root = (root or settings.media_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # --- key helpers --------------------------------------------------------
    @staticmethod
    def build_key(category: str, filename: str) -> str:
        """``uploads/2026/09/3f2a…-recording.wav``"""
        stamp = datetime.now(UTC)
        safe_name = sanitize_filename(filename)
        return f"{category}/{stamp:%Y/%m}/{uuid4().hex[:12]}-{safe_name}"

    def path_for(self, key: str) -> Path:
        """Resolve a storage key to an absolute path, refusing traversal."""
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root):
            raise StorageError(f"Refusing to resolve key outside the storage root: {key!r}")
        return candidate

    def exists(self, key: str) -> bool:
        return self.path_for(key).exists()

    def reserve(self, category: str, filename: str) -> tuple[str, Path]:
        """Allocate a key and make sure its directory exists."""
        key = self.build_key(category, filename)
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return key, path

    # --- writes -------------------------------------------------------------
    def save_stream(self, stream, category: str, filename: str) -> tuple[str, Path]:
        """Persist a Werkzeug ``FileStorage`` (or any object with ``save``/``read``)."""
        key, path = self.reserve(category, filename)
        if hasattr(stream, "save"):
            stream.save(path)
        else:  # pragma: no cover - defensive branch for plain file objects
            path.write_bytes(stream.read())
        return key, path

    def import_file(self, source: Path, category: str, filename: str | None = None) -> tuple[str, Path]:
        key, path = self.reserve(category, filename or source.name)
        shutil.copy2(source, path)
        return key, path

    def delete(self, key: str | None) -> None:
        if not key:
            return
        try:
            self.path_for(key).unlink(missing_ok=True)
        except StorageError:
            logger.warning("refused to delete suspicious key %r", key)

    # --- housekeeping -------------------------------------------------------
    def prune(self, category: str, keep: int) -> int:
        """Delete all but the ``keep`` most recent files in a category."""
        if keep <= 0:
            return 0
        directory = self.root / category
        if not directory.exists():
            return 0
        files = sorted(
            (path for path in directory.rglob("*") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        removed = 0
        for path in files[keep:]:
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def usage_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())


def sanitize_filename(filename: str) -> str:
    """Strip directories and unsafe characters from a user-supplied filename."""
    from werkzeug.utils import secure_filename

    cleaned = secure_filename(Path(filename).name) or "audio"
    return cleaned[:96]


def get_storage(settings: Settings | None = None) -> MediaStorage:
    return MediaStorage(settings=settings)
