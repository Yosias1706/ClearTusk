from pathlib import Path

from cleartusk.config import Settings, find_project_root


def test_paths_derive_from_data_dir(tmp_path):
    settings = Settings(data_dir=tmp_path / "corpusroot")
    assert settings.resolved_corpus_dir == (tmp_path / "corpusroot" / "corpus").resolve()
    assert settings.call_clips_dir.name == "calls"
    assert settings.detector_model_path.parent == settings.models_dir


def test_runtime_dir_can_be_relocated_independently(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", runtime_dir=tmp_path / "scratch")
    assert settings.media_dir == (tmp_path / "scratch" / "media").resolve()
    assert settings.resolved_data_dir != settings.resolved_runtime_dir


def test_database_url_defaults_to_sqlite_inside_runtime(tmp_path):
    settings = Settings(data_dir=tmp_path)
    assert settings.resolved_database_url.startswith("sqlite:///")
    assert settings.database_backend == "sqlite"


def test_postgres_url_is_reported_as_such(tmp_path):
    settings = Settings(database_url="postgresql+psycopg://user:pw@db:5432/cleartusk")
    assert settings.database_backend == "postgresql"


def test_extensions_accept_a_comma_string():
    settings = Settings(allowed_extensions="wav, .flac ,mp3")
    assert settings.allowed_extensions == (".wav", ".flac", ".mp3")


def test_project_root_marker_is_found():
    assert (find_project_root() / "pyproject.toml").exists()


def test_upload_limit_converts_to_bytes():
    assert Settings(max_upload_mb=8).max_upload_bytes == 8 * 1024 * 1024


def test_no_module_hardcodes_an_absolute_user_path():
    package = Path(__file__).resolve().parents[1] / "cleartusk"
    offenders = [
        path
        for path in package.rglob("*.py")
        if "/Users/" in path.read_text(encoding="utf-8") or "C:\\\\" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
