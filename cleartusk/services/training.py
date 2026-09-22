"""Detector training workflow.

Positive examples are the annotated call clips; negative examples are the
call-free "safe noise" windows cut from the *same* recordings, so the model has
to learn what a call sounds like rather than which field site it came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import update
from sqlalchemy.orm import Session

from cleartusk.audio.detection import ALGORITHM, MODEL_NAME, TrainingReport, save_model, train_detector
from cleartusk.audio.features import FEATURE_NAMES
from cleartusk.audio.io import load_analysis_audio
from cleartusk.config import Settings, get_settings
from cleartusk.db.models import ModelVersion
from cleartusk.services.corpus import clip_source_map

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingMatrix:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray

    def __len__(self) -> int:
        return int(self.features.shape[0])


def iter_windows(
    audio: np.ndarray, sample_rate: int, window_seconds: float, max_windows: int = 6
) -> list[np.ndarray]:
    """Slice a clip into the same fixed-length windows the detector sees at inference time.

    Training on whole clips would leak clip *length* into the model, because the
    noise clips are cut from gaps and the call clips from annotations. Windowing
    removes that shortcut and multiplies the effective sample count.
    """
    window = max(int(window_seconds * sample_rate), 1)
    if audio.size <= window:
        return [audio]

    hop = max(window // 2, 1)
    offsets = list(range(0, audio.size - window + 1, hop))
    if len(offsets) > max_windows:
        # Evenly spread the retained windows across the clip.
        picks = np.linspace(0, len(offsets) - 1, max_windows).round().astype(int)
        offsets = [offsets[index] for index in dict.fromkeys(picks.tolist())]
    return [audio[offset : offset + window] for offset in offsets]


def build_training_matrix(settings: Settings | None = None) -> TrainingMatrix:
    """Featurise windows from every call clip (label 1) and noise clip (label 0)."""
    from cleartusk.audio.features import extract_features  # local import keeps module import cheap

    settings = settings or get_settings()
    sources = clip_source_map(settings)

    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    def add(path, label: int, clip_name: str) -> None:
        audio = load_analysis_audio(path)
        if audio.size == 0:
            return
        group = sources.get(clip_name, clip_name)
        for window in iter_windows(audio, settings.sample_rate, settings.detector_window_seconds):
            rows.append(extract_features(window, settings.sample_rate))
            labels.append(label)
            groups.append(group)

    for path in sorted(settings.call_clips_dir.glob("*.wav")):
        add(path, 1, path.stem)
    for path in sorted(settings.noise_clips_dir.glob("*_noise.wav")):
        add(path, 0, path.stem.removesuffix("_noise"))

    if not rows:
        raise RuntimeError(
            "No training clips found. Run `cleartusk prepare-corpus` first to cut clips from raw audio."
        )

    return TrainingMatrix(
        features=np.vstack(rows),
        labels=np.asarray(labels, dtype=int),
        groups=np.asarray(groups, dtype=object),
    )


def register_model(session: Session, report: TrainingReport, artifact_key: str) -> ModelVersion:
    """Persist the scorecard and mark the new checkpoint active."""
    session.execute(update(ModelVersion).where(ModelVersion.name == MODEL_NAME).values(is_active=False))
    version = ModelVersion(
        name=MODEL_NAME,
        algorithm=report.algorithm,
        artifact_key=artifact_key,
        feature_count=report.feature_count,
        sample_count=report.sample_count,
        positive_count=report.positive_count,
        negative_count=report.negative_count,
        cv_folds=report.cv_folds,
        cv_strategy=report.cv_strategy,
        accuracy=report.accuracy,
        precision=report.precision,
        recall=report.recall,
        f1=report.f1,
        roc_auc=report.roc_auc,
        is_active=True,
        details={
            "fold_scores": report.fold_scores,
            "top_features": report.top_features,
            "feature_names": list(FEATURE_NAMES),
        },
    )
    session.add(version)
    session.flush()
    return version


def train_and_register(session: Session, folds: int = 5, settings: Settings | None = None) -> TrainingReport:
    """Full training run: featurise, cross-validate, refit, persist, record."""
    settings = settings or get_settings()
    matrix = build_training_matrix(settings)
    logger.info(
        "training %s on %d samples (%d call / %d noise)",
        ALGORITHM,
        len(matrix),
        int(matrix.labels.sum()),
        int(len(matrix) - matrix.labels.sum()),
    )
    model, report = train_detector(matrix.features, matrix.labels, matrix.groups, folds=folds)
    artifact_path = save_model(model, settings.detector_model_path)
    register_model(session, report, _display_path(artifact_path, settings))
    return report


def _display_path(path: Path, settings: Settings) -> str:
    """Project-relative when possible, absolute otherwise (runtime dirs may live anywhere)."""
    try:
        return str(path.relative_to(settings.project_root))
    except ValueError:
        return str(path)
