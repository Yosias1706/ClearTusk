import pytest
from sqlalchemy import select

from cleartusk.db.models import CorpusClip, ProcessingRun, Recording
from cleartusk.services.analytics import AnalyticsService, histogram
from cleartusk.services.benchmark import run_benchmark, summarize
from cleartusk.services.corpus import classify_noise_source, ingest_corpus, load_annotations
from cleartusk.services.demo import build_showcase
from cleartusk.services.pipeline import CleaningPipeline, PipelineError
from cleartusk.services.storage import MediaStorage, StorageError
from tests.conftest import EXPECTED_BACKEND


def test_annotations_load_with_derived_fields(corpus):
    annotations = load_annotations(corpus)
    assert len(annotations) == 6
    first = annotations[0]
    assert first.clip_name.startswith("call_")
    assert first.duration_seconds == pytest.approx(3.0)
    assert first.call_type == "rumble"
    assert first.noise_source == "vehicle"


def test_noise_source_is_inferred_from_the_filename():
    assert classify_noise_source("99-45_airplane_01.wav") == "airplane"
    assert classify_noise_source("2000-24_generator_noise_4.wav") == "generator"
    assert classify_noise_source("mystery.wav") == "unknown"


def test_corpus_ingest_is_idempotent(corpus, session):
    assert ingest_corpus(session, corpus) == 6
    assert ingest_corpus(session, corpus) == 6
    assert session.scalar(select(CorpusClip).where(CorpusClip.clip_name == "call_1")).has_noise_reference


def test_storage_refuses_path_traversal(settings):
    storage = MediaStorage(settings=settings)
    with pytest.raises(StorageError):
        storage.path_for("../../etc/passwd")


def test_storage_prunes_to_the_retention_limit(settings):
    storage = MediaStorage(settings=settings)
    for index in range(5):
        key, path = storage.reserve("uploads", f"file{index}.wav")
        path.write_bytes(b"x")
    assert storage.prune("uploads", keep=2) == 3
    assert len(list((storage.root / "uploads").rglob("*.wav"))) == 2


def test_pipeline_persists_a_full_run(corpus, session, wav_upload):
    outcome = CleaningPipeline(session).process_file(wav_upload, "rumble")

    run = session.scalar(select(ProcessingRun).where(ProcessingRun.public_id == outcome.run_public_id))
    assert run is not None
    assert run.status == "succeeded"
    assert run.metrics is not None
    assert run.resolved_call_type == "rumble"
    assert run.call_type_source == "user"
    assert run.realtime_factor > 0
    assert session.scalar(select(Recording).where(Recording.id == run.recording_id)).content_hash

    storage = MediaStorage(settings=corpus)
    for key in (outcome.cleaned_key, outcome.original_spectrogram_key, outcome.comparison_spectrogram_key):
        assert storage.exists(key)


def test_pipeline_auto_selects_a_profile(corpus, session, wav_upload):
    outcome = CleaningPipeline(session).process_file(wav_upload, "auto")
    assert outcome.call_type_source == "auto"
    assert outcome.call_type in {"rumble", "roar", "trumpet", "default"}


def test_pipeline_rejects_unsupported_uploads(corpus, session, tmp_path):
    class FakeUpload:
        filename = "notes.txt"

        def save(self, path):  # pragma: no cover - never reached
            path.write_bytes(b"")

    with pytest.raises(PipelineError):
        CleaningPipeline(session).process_upload(FakeUpload(), "auto")


def test_benchmark_records_aggregates(corpus, session):
    result = run_benchmark(session, label="test sweep", workers=1, limit=4)
    assert result.clip_count == 4
    assert len(result.results) == 4
    assert result.realtime_factor > 0
    metrics = result.aggregates["metrics"]
    assert set(metrics) >= {"target_retention_pct", "machine_suppression_pct"}
    assert result.aggregates["preset_distribution"]


def test_summarize_handles_an_empty_sweep():
    assert summarize([], 1.0, 1) == {}


def test_histogram_bins_values():
    result = histogram([1, 5, 9], [0, 5, 10])
    assert result.counts == [1, 2]
    assert len(result.labels) == 2


def test_analytics_reports_runs_and_corpus(corpus, session, wav_upload):
    ingest_corpus(session, corpus)
    CleaningPipeline(session).process_file(wav_upload, "rumble")
    run_benchmark(session, workers=1, limit=3)

    data = AnalyticsService(session, corpus).dashboard()
    assert data["overview"]["runs_total"] == 1
    assert data["overview"]["corpus_clips"] == 6
    assert data["overview"]["avg_target_retention_pct"] > 0
    assert data["performance"]["latency_p50_ms"] > 0
    assert data["quality_by_call_type"]
    assert data["recent_runs"][0]["filename"] == wav_upload.name
    assert data["storage"]["database_backend"] == EXPECTED_BACKEND


def test_analytics_is_safe_on_an_empty_database(session, settings):
    data = AnalyticsService(session, settings).dashboard()
    assert data["overview"]["runs_total"] == 0
    assert data["detector"] is None
    assert data["quality_by_call_type"] == []


def test_showcase_renders_once_and_is_cached(corpus):
    first = build_showcase(corpus)
    assert first.available
    storage = MediaStorage(settings=corpus)
    assert storage.exists(first.comparison_key)

    stamp = storage.path_for(first.comparison_key).stat().st_mtime_ns
    second = build_showcase(corpus)
    assert storage.path_for(second.comparison_key).stat().st_mtime_ns == stamp


def test_showcase_reports_unavailable_without_a_corpus(settings):
    assert build_showcase(settings).available is False
