"""Call-activity detection.

A gradient-boosted classifier separates windows that contain an elephant
vocalisation from windows of machine/background noise taken from the *same*
recordings. Cross-validation is grouped by source recording, so a model cannot
score well by memorising the acoustic signature of one field site.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cleartusk.audio.features import FEATURE_NAMES, extract_features, infer_call_type
from cleartusk.config import get_settings

logger = logging.getLogger(__name__)

ALGORITHM = "HistGradientBoostingClassifier"
MODEL_NAME = "call_detector"


@dataclass(frozen=True, slots=True)
class DetectedEvent:
    """A contiguous stretch of audio classified as containing a call."""

    start_seconds: float
    end_seconds: float
    confidence: float
    peak_frequency_hz: float
    predicted_call_type: str

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def as_dict(self) -> dict[str, float | str]:
        return {**asdict(self), "duration_seconds": self.duration_seconds}


@dataclass(frozen=True, slots=True)
class DetectionSummary:
    """Result of sweeping a detector over one recording."""

    events: tuple[DetectedEvent, ...]
    peak_confidence: float
    coverage_pct: float
    model_name: str
    timeline: tuple[tuple[float, float], ...] = field(default=())

    @property
    def call_present(self) -> bool:
        return bool(self.events)

    def dominant_call_type(self) -> str | None:
        if not self.events:
            return None
        return max(
            self.events, key=lambda event: event.confidence * event.duration_seconds
        ).predicted_call_type

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_present": self.call_present,
            "event_count": len(self.events),
            "peak_confidence": self.peak_confidence,
            "coverage_pct": self.coverage_pct,
            "model_name": self.model_name,
            "dominant_call_type": self.dominant_call_type(),
            "events": [event.as_dict() for event in self.events],
            "timeline": [{"t": t, "p": p} for t, p in self.timeline],
        }


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """Cross-validated scorecard for a freshly trained detector."""

    algorithm: str
    feature_count: int
    sample_count: int
    positive_count: int
    negative_count: int
    cv_folds: int
    cv_strategy: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    fold_scores: list[dict[str, float]]
    top_features: list[dict[str, float | str]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_estimator(random_state: int = 7) -> Pipeline:
    """Scaler + gradient boosting; scaling keeps the pipeline swappable for linear models."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.06,
                    max_depth=4,
                    min_samples_leaf=8,
                    l2_regularization=0.5,
                    early_stopping=False,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _permutation_importance(model: Pipeline, features: np.ndarray, labels: np.ndarray, top_n: int = 8):
    """Cheap permutation importance over the training matrix."""
    rng = np.random.default_rng(7)
    baseline = accuracy_score(labels, model.predict(features))
    names = (
        FEATURE_NAMES
        if features.shape[1] == len(FEATURE_NAMES)
        else tuple(f"feature_{index}" for index in range(features.shape[1]))
    )
    drops: list[tuple[str, float]] = []
    for index, name in enumerate(names):
        shuffled = features.copy()
        rng.shuffle(shuffled[:, index])
        drops.append((name, float(baseline - accuracy_score(labels, model.predict(shuffled)))))
    drops.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "accuracy_drop": round(drop, 4)} for name, drop in drops[:top_n]]


def cross_validate(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int = 5,
) -> tuple[dict[str, float], list[dict[str, float]], str]:
    """Grouped stratified cross-validation; falls back to stratified folds for tiny corpora."""
    unique_groups = len(set(groups.tolist()))
    folds = max(2, min(folds, unique_groups, int(np.bincount(labels).min())))
    strategy = f"StratifiedGroupKFold(n_splits={folds}, groups=source_recording)"
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=7)

    fold_scores: list[dict[str, float]] = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(features, labels, groups), start=1):
        model = build_estimator()
        model.fit(features[train_index], labels[train_index])
        predictions = model.predict(features[test_index])
        probabilities = model.predict_proba(features[test_index])[:, 1]
        truth = labels[test_index]
        fold_scores.append(
            {
                "fold": fold_index,
                "n_test": int(len(test_index)),
                "accuracy": float(accuracy_score(truth, predictions)),
                "precision": float(precision_score(truth, predictions, zero_division=0)),
                "recall": float(recall_score(truth, predictions, zero_division=0)),
                "f1": float(f1_score(truth, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(truth, probabilities))
                if len(set(truth)) > 1
                else float("nan"),
            }
        )

    keys = ("accuracy", "precision", "recall", "f1", "roc_auc")
    means: dict[str, float] = {}
    for key in keys:
        values = [score[key] for score in fold_scores if not np.isnan(score[key])]
        means[key] = float(np.mean(values)) if values else 0.0
    return means, fold_scores, strategy


def train_detector(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int = 5,
) -> tuple[Pipeline, TrainingReport]:
    """Cross-validate, then refit on all data and return the deployable model."""
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must align on the first axis")
    if len(set(labels.tolist())) < 2:
        raise ValueError("training needs both call and non-call examples")

    means, fold_scores, strategy = cross_validate(features, labels, groups, folds)
    model = build_estimator()
    model.fit(features, labels)

    report = TrainingReport(
        algorithm=ALGORITHM,
        feature_count=int(features.shape[1]),
        sample_count=int(features.shape[0]),
        positive_count=int(labels.sum()),
        negative_count=int(len(labels) - labels.sum()),
        cv_folds=len(fold_scores),
        cv_strategy=strategy,
        accuracy=means["accuracy"],
        precision=means["precision"],
        recall=means["recall"],
        f1=means["f1"],
        roc_auc=means["roc_auc"],
        fold_scores=fold_scores,
        top_features=_permutation_importance(model, features, labels),
    )
    return model, report


def save_model(model: Pipeline, path: Path | None = None) -> Path:
    settings = get_settings()
    path = path or settings.detector_model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": list(FEATURE_NAMES)}, path)
    return path


class CallDetector:
    """Sliding-window wrapper around the trained classifier."""

    def __init__(self, model: Pipeline | None, model_name: str = MODEL_NAME) -> None:
        self._model = model
        self.model_name = model_name

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @classmethod
    def load(cls, path: Path | None = None) -> CallDetector:
        settings = get_settings()
        path = path or settings.detector_model_path
        if not path.exists():
            logger.info("no detector checkpoint at %s; detection disabled", path)
            return cls(None)
        payload = joblib.load(path)
        if list(payload.get("feature_names", [])) != list(FEATURE_NAMES):
            logger.warning("detector checkpoint was trained on a different feature set; ignoring it")
            return cls(None)
        return cls(payload["model"], model_name=path.stem)

    def score_windows(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(window_start_times, call_probabilities)``."""
        if self._model is None or audio.size == 0:
            return np.zeros(0), np.zeros(0)

        settings = get_settings()
        window = max(int(settings.detector_window_seconds * sample_rate), 1)
        hop = max(int(settings.detector_hop_seconds * sample_rate), 1)

        if audio.size <= window:
            starts = np.zeros(1)
            matrix = extract_features(audio, sample_rate)[None, :]
        else:
            offsets = list(range(0, audio.size - window + 1, hop))
            starts = np.asarray([offset / sample_rate for offset in offsets], dtype=float)
            matrix = np.vstack(
                [extract_features(audio[offset : offset + window], sample_rate) for offset in offsets]
            )
        return starts, self._model.predict_proba(matrix)[:, 1]

    def detect(self, audio: np.ndarray, sample_rate: int) -> DetectionSummary:
        """Sweep the recording and merge above-threshold windows into events."""
        settings = get_settings()
        starts, probabilities = self.score_windows(audio, sample_rate)
        if starts.size == 0:
            return DetectionSummary((), 0.0, 0.0, self.model_name)

        window_seconds = min(settings.detector_window_seconds, audio.size / sample_rate)
        above = probabilities >= settings.detector_threshold

        events: list[DetectedEvent] = []
        index = 0
        while index < len(above):
            if not above[index]:
                index += 1
                continue
            end_index = index
            while end_index + 1 < len(above) and above[end_index + 1]:
                end_index += 1

            start_seconds = float(starts[index])
            end_seconds = float(min(starts[end_index] + window_seconds, audio.size / sample_rate))
            confidence = float(probabilities[index : end_index + 1].max())
            if end_seconds - start_seconds >= settings.detector_min_event_seconds:
                segment = audio[int(start_seconds * sample_rate) : int(end_seconds * sample_rate)]
                call_type, _ = infer_call_type(segment, sample_rate)
                freqs = np.fft.rfftfreq(max(len(segment), 2), 1 / sample_rate)
                spectrum = np.abs(np.fft.rfft(segment)) if segment.size else np.zeros(1)
                peak_frequency = float(freqs[int(np.argmax(spectrum))]) if spectrum.size > 1 else 0.0
                events.append(
                    DetectedEvent(
                        start_seconds=round(start_seconds, 3),
                        end_seconds=round(end_seconds, 3),
                        confidence=round(confidence, 4),
                        peak_frequency_hz=round(peak_frequency, 2),
                        predicted_call_type=call_type,
                    )
                )
            index = end_index + 1

        total_seconds = audio.size / sample_rate
        covered = sum(event.duration_seconds for event in events)
        timeline = tuple(
            (round(float(start), 3), round(float(probability), 4))
            for start, probability in zip(starts, probabilities, strict=True)
        )
        return DetectionSummary(
            events=tuple(events),
            peak_confidence=float(probabilities.max()),
            coverage_pct=float(covered / total_seconds * 100) if total_seconds else 0.0,
            model_name=self.model_name,
            timeline=timeline,
        )
