"""Corpus preparation and ingestion.

The reference corpus is 38 field recordings annotated with 212 elephant calls.
This module turns that annotation table into three aligned clip sets and keeps
the database in sync with what is on disk:

``calls/``    the annotated vocalisation
``context/``  the same call plus a two-second margin either side
``noise/``    the nearest gap without a call, used as a per-recording noise reference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from cleartusk.audio.presets import normalize_call_type
from cleartusk.config import Settings, get_settings
from cleartusk.db.models import CorpusClip

logger = logging.getLogger(__name__)

CONTEXT_BUFFER_MS = 2000
MIN_NOISE_MS = 1500

#: Machine-noise categories recoverable from the source filenames.
NOISE_CATEGORIES = ("airplane", "vehicle", "generator", "background")


@dataclass(frozen=True, slots=True)
class Annotation:
    """One row of the annotation table."""

    selection: int
    source_file: str
    start_seconds: float
    end_seconds: float
    raw_call_type: str

    @property
    def clip_name(self) -> str:
        return f"call_{self.selection}"

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    @property
    def call_type(self) -> str:
        return normalize_call_type(self.raw_call_type)

    @property
    def noise_source(self) -> str:
        return classify_noise_source(self.source_file)


def classify_noise_source(source_file: str) -> str:
    """Infer the dominant interference type from a source recording's name."""
    lowered = source_file.lower()
    for category in NOISE_CATEGORIES:
        if category in lowered:
            return category
    return "unknown"


def load_annotations(settings: Settings | None = None) -> list[Annotation]:
    """Read and validate ``annotations.csv``."""
    settings = settings or get_settings()
    path = settings.annotations_csv
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    frame = pd.read_csv(path)
    required = {"Selection", "Sound_file", "Start_time", "End_time", "Call_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame.dropna(subset=["Selection", "Sound_file", "Start_time", "End_time"])
    annotations = [
        Annotation(
            selection=int(row.Selection),
            source_file=str(row.Sound_file).strip(),
            start_seconds=float(row.Start_time),
            end_seconds=float(row.End_time),
            raw_call_type=str(row.Call_type).strip(),
        )
        for row in frame.itertuples(index=False)
    ]
    return sorted(annotations, key=lambda item: (item.source_file, item.start_seconds))


def clip_source_map(settings: Settings | None = None) -> dict[str, str]:
    """Map ``call_12`` to the recording it was cut from (used to group CV folds)."""
    try:
        return {item.clip_name: item.source_file for item in load_annotations(settings)}
    except FileNotFoundError:
        return {}


def prepare_clips(settings: Settings | None = None, overwrite: bool = False) -> dict[str, int]:
    """Cut call, context, and safe-noise clips out of the raw recordings."""
    from pydub import AudioSegment  # imported lazily: only needed for corpus preparation

    settings = settings or get_settings()
    annotations = load_annotations(settings)
    for directory in (settings.call_clips_dir, settings.context_clips_dir, settings.noise_clips_dir):
        directory.mkdir(parents=True, exist_ok=True)

    counts = {"calls": 0, "context": 0, "noise": 0, "skipped": 0}
    by_source: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        by_source.setdefault(annotation.source_file, []).append(annotation)

    def export(audio, start_ms: int, end_ms: int, destination: Path) -> bool:
        if end_ms <= start_ms:
            return False
        if destination.exists() and not overwrite:
            return True
        audio[start_ms:end_ms].export(destination, format="wav")
        return True

    for source_file, items in by_source.items():
        source_path = settings.raw_audio_dir / source_file
        if not source_path.exists():
            logger.warning("raw recording missing, skipping: %s", source_path)
            counts["skipped"] += len(items)
            continue

        audio = AudioSegment.from_file(source_path)
        length_ms = len(audio)

        for index, annotation in enumerate(items):
            start_ms = max(0, int(round(annotation.start_seconds * 1000)))
            end_ms = min(length_ms, int(round(annotation.end_seconds * 1000)))
            if end_ms <= start_ms:
                counts["skipped"] += 1
                continue

            if export(audio, start_ms, end_ms, settings.call_clips_dir / f"{annotation.clip_name}.wav"):
                counts["calls"] += 1
            if export(
                audio,
                max(0, start_ms - CONTEXT_BUFFER_MS),
                min(length_ms, end_ms + CONTEXT_BUFFER_MS),
                settings.context_clips_dir / f"{annotation.clip_name}_context.wav",
            ):
                counts["context"] += 1

            previous_end = 0 if index == 0 else int(round(items[index - 1].end_seconds * 1000))
            next_start = (
                length_ms if index == len(items) - 1 else int(round(items[index + 1].start_seconds * 1000))
            )
            window = _safe_noise_window(start_ms, end_ms, previous_end, next_start)
            if window and export(
                audio, *window, settings.noise_clips_dir / f"{annotation.clip_name}_noise.wav"
            ):
                counts["noise"] += 1

    logger.info("corpus prepared: %s", counts)
    return counts


def _safe_noise_window(
    start_ms: int, end_ms: int, previous_end_ms: int, next_start_ms: int
) -> tuple[int, int] | None:
    """Pick the widest call-free gap adjacent to this call."""
    left_gap = start_ms - previous_end_ms
    right_gap = next_start_ms - end_ms
    if left_gap >= MIN_NOISE_MS and left_gap >= right_gap:
        return previous_end_ms, start_ms
    if right_gap >= MIN_NOISE_MS:
        return end_ms, next_start_ms
    return None


def ingest_corpus(session: Session, settings: Settings | None = None) -> int:
    """Replace the ``corpus_clips`` table with the current annotation state."""
    settings = settings or get_settings()
    annotations = load_annotations(settings)

    session.execute(delete(CorpusClip))
    session.add_all(
        CorpusClip(
            selection=annotation.selection,
            clip_name=annotation.clip_name,
            source_file=annotation.source_file,
            noise_source=annotation.noise_source,
            raw_call_type=annotation.raw_call_type,
            call_type=annotation.call_type,
            start_seconds=annotation.start_seconds,
            end_seconds=annotation.end_seconds,
            duration_seconds=annotation.duration_seconds,
            has_noise_reference=(settings.noise_clips_dir / f"{annotation.clip_name}_noise.wav").exists(),
            has_context=(settings.context_clips_dir / f"{annotation.clip_name}_context.wav").exists(),
        )
        for annotation in annotations
    )
    session.flush()
    return len(annotations)


def corpus_clip_paths(settings: Settings | None = None) -> list[Path]:
    """Every prepared call clip, in stable order."""
    settings = settings or get_settings()
    return sorted(settings.call_clips_dir.glob("*.wav"), key=lambda path: path.stem)


def noise_reference_path(clip_name: str, settings: Settings | None = None) -> Path | None:
    settings = settings or get_settings()
    path = settings.noise_clips_dir / f"{clip_name}_noise.wav"
    return path if path.exists() else None


def context_path(clip_name: str, settings: Settings | None = None) -> Path | None:
    settings = settings or get_settings()
    path = settings.context_clips_dir / f"{clip_name}_context.wav"
    return path if path.exists() else None


def corpus_is_present(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return settings.annotations_csv.exists() and any(settings.call_clips_dir.glob("*.wav"))
