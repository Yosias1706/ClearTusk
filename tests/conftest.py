"""Shared fixtures.

Tests run against a throwaway SQLite database and a synthetic corpus written
into ``tmp_path``, so they never touch the real data directory.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf

from cleartusk.config import reload_settings
from cleartusk.db import init_database, reset_engine, session_scope

SAMPLE_RATE = 2000

#: Point this at a PostgreSQL server to run the identical suite against it.
#: CI does exactly that, so the SQLite fallback and PostgreSQL cannot drift.
TEST_DATABASE_URL = os.environ.get("CLEARTUSK_TEST_DATABASE_URL")
EXPECTED_BACKEND = (TEST_DATABASE_URL or "sqlite").split(":", 1)[0].split("+", 1)[0]


def synth_call(seconds: float = 3.0, fundamental: float = 22.0, seed: int = 0) -> np.ndarray:
    """A synthetic rumble: low fundamental plus harmonics under an envelope."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * SAMPLE_RATE), endpoint=False)
    envelope = np.hanning(t.size)
    tone = sum((1.0 / (index + 1)) * np.sin(2 * np.pi * fundamental * (index + 1) * t) for index in range(4))
    engine = 0.35 * np.sin(2 * np.pi * 320 * t) + 0.25 * np.sin(2 * np.pi * 480 * t)
    hiss = 0.05 * rng.standard_normal(t.size)
    return ((tone * envelope) + engine + hiss).astype(np.float32)


def synth_noise(seconds: float = 3.0, seed: int = 1) -> np.ndarray:
    """Machine noise only: steady tones plus broadband hiss, no call."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * SAMPLE_RATE), endpoint=False)
    engine = 0.4 * np.sin(2 * np.pi * 320 * t) + 0.3 * np.sin(2 * np.pi * 480 * t)
    return (engine + 0.08 * rng.standard_normal(t.size)).astype(np.float32)


def write_wav(path, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sample_rate, subtype="PCM_16")


@pytest.fixture(autouse=True)
def ignore_local_dotenv():
    """Never read a developer's .env: tests must be reproducible everywhere."""
    from cleartusk.config import Settings

    original = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None
    yield
    Settings.model_config["env_file"] = original


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Isolated settings backed by a temporary data directory."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("CLEARTUSK_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CLEARTUSK_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("CLEARTUSK_DATABASE_URL", TEST_DATABASE_URL or f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("CLEARTUSK_SECRET_KEY", "test-secret")
    monkeypatch.setenv("CLEARTUSK_RETAIN_RUN_ARTIFACTS", "50")

    reset_engine()
    current = reload_settings()
    current.ensure_runtime_dirs()
    yield current
    reset_engine()
    reload_settings()


@pytest.fixture
def corpus(settings):
    """Six call clips and six noise clips across three 'source recordings'."""
    rows = ["Selection,Sound_file,Start_time,End_time,Call_type"]
    for index in range(6):
        clip = f"call_{index + 1}"
        source = f"site{index % 3}_vehicle_01.wav"
        write_wav(settings.call_clips_dir / f"{clip}.wav", synth_call(seed=index, fundamental=20 + index))
        write_wav(settings.noise_clips_dir / f"{clip}_noise.wav", synth_noise(seed=100 + index))
        write_wav(settings.context_clips_dir / f"{clip}_context.wav", synth_call(seconds=5.0, seed=index))
        rows.append(f"{index + 1},{source},0.0,3.0,rumble")

    settings.annotations_csv.parent.mkdir(parents=True, exist_ok=True)
    settings.annotations_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return settings


@pytest.fixture
def database(settings):
    init_database(drop=True)
    return settings


@pytest.fixture
def session(database):
    with session_scope() as db_session:
        yield db_session


@pytest.fixture
def app(database):
    from cleartusk.web import create_app

    application = create_app(database)
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def wav_upload(tmp_path):
    """A small on-disk WAV suitable for uploading."""
    path = tmp_path / "field-recording.wav"
    write_wav(path, synth_call(seconds=4.0), SAMPLE_RATE)
    return path
