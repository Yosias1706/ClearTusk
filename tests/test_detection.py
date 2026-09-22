import numpy as np

from cleartusk.audio.detection import CallDetector, save_model, train_detector
from cleartusk.services.training import build_training_matrix, iter_windows, train_and_register
from tests.conftest import SAMPLE_RATE, synth_call, synth_noise


def test_windows_match_the_inference_window(settings):
    windows = iter_windows(synth_call(seconds=6.0), SAMPLE_RATE, 1.5)
    assert len(windows) > 1
    assert all(window.size == int(1.5 * SAMPLE_RATE) for window in windows)


def test_short_clips_yield_a_single_window(settings):
    windows = iter_windows(synth_call(seconds=1.0), SAMPLE_RATE, 1.5)
    assert len(windows) == 1


def test_training_matrix_labels_calls_and_noise(corpus):
    matrix = build_training_matrix(corpus)
    assert matrix.features.shape[0] == matrix.labels.shape[0] == matrix.groups.shape[0]
    assert set(matrix.labels.tolist()) == {0, 1}
    # Folds are grouped by source recording, so groups must not be unique per clip.
    assert len(set(matrix.groups.tolist())) < matrix.features.shape[0]


def test_detector_trains_and_registers_a_scorecard(corpus, session):
    report = train_and_register(session, folds=3)
    assert 0.0 <= report.roc_auc <= 1.0
    assert report.sample_count > 0
    assert report.cv_folds >= 2
    assert "StratifiedGroupKFold" in report.cv_strategy
    assert corpus.detector_model_path.exists()


def test_detector_finds_a_call_embedded_in_noise(corpus, session):
    train_and_register(session, folds=3)
    detector = CallDetector.load(corpus.detector_model_path)
    assert detector.is_available

    silence = synth_noise(seconds=3.0, seed=7)
    recording = np.concatenate([silence, synth_call(seconds=3.0, seed=3), silence])
    summary = detector.detect(recording, SAMPLE_RATE)

    assert summary.peak_confidence > 0.5
    assert summary.timeline
    if summary.events:
        best = max(summary.events, key=lambda event: event.confidence)
        assert 1.0 < best.start_seconds < 6.0


def test_missing_checkpoint_disables_detection_gracefully(settings):
    detector = CallDetector.load(settings.models_dir / "does-not-exist.joblib")
    assert not detector.is_available
    summary = detector.detect(synth_call(), SAMPLE_RATE)
    assert summary.events == ()
    assert summary.as_dict()["call_present"] is False


def test_checkpoint_with_a_stale_feature_set_is_ignored(settings, monkeypatch):
    import joblib

    from cleartusk.audio import detection as detection_module

    features = np.random.default_rng(0).random((20, 4))
    labels = np.array([0, 1] * 10)
    groups = np.array(["a", "b"] * 10)
    model, _ = train_detector(features, labels, groups, folds=2)
    path = settings.models_dir / "stale.joblib"
    joblib.dump({"model": model, "feature_names": ["a", "b", "c", "d"]}, path)

    assert not detection_module.CallDetector.load(path).is_available


def test_save_model_creates_parent_directories(settings):
    features = np.random.default_rng(1).random((20, 4))
    labels = np.array([0, 1] * 10)
    groups = np.array(["x", "y"] * 10)
    model, _ = train_detector(features, labels, groups, folds=2)
    path = save_model(model, settings.models_dir / "nested" / "model.joblib")
    assert path.exists()
