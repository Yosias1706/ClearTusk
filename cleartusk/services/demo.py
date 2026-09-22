"""Landing-page showcase.

Renders a cached before/after example straight from the corpus so the home page
always shows real output from the current DSP rather than a checked-in
screenshot. Artifacts are written once under ``media/demo`` and reused.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cleartusk.audio.denoise import clean_signal
from cleartusk.audio.io import load_analysis_audio, resample, write_audio
from cleartusk.audio.presets import get_profile
from cleartusk.audio.spectrogram import render_comparison
from cleartusk.config import Settings, get_settings
from cleartusk.services.corpus import load_annotations, noise_reference_path
from cleartusk.services.storage import MediaStorage

logger = logging.getLogger(__name__)

DEMO_PREFIX = "demo"
PREFERRED_CLIPS = ("call_4", "call_108", "call_3")


@dataclass(frozen=True, slots=True)
class Showcase:
    """Cached demo artifacts plus the metrics they achieved."""

    available: bool
    clip_name: str = ""
    call_type: str = ""
    source_file: str = ""
    noise_source: str = ""
    original_audio_key: str = ""
    cleaned_audio_key: str = ""
    comparison_key: str = ""
    metrics: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "clip_name": self.clip_name,
            "call_type": self.call_type,
            "source_file": self.source_file,
            "noise_source": self.noise_source,
            "metrics": self.metrics or {},
        }


def _pick_clip(settings: Settings) -> str | None:
    for name in PREFERRED_CLIPS:
        if (settings.call_clips_dir / f"{name}.wav").exists():
            return name
    clips = sorted(settings.call_clips_dir.glob("*.wav"))
    return clips[0].stem if clips else None


def build_showcase(settings: Settings | None = None, force: bool = False) -> Showcase:
    """Return the cached showcase, rendering it on first use."""
    settings = settings or get_settings()
    clip_name = _pick_clip(settings)
    if clip_name is None:
        return Showcase(available=False)

    storage = MediaStorage(settings=settings)
    keys = {
        "original": f"{DEMO_PREFIX}/{clip_name}-original.wav",
        "cleaned": f"{DEMO_PREFIX}/{clip_name}-cleaned.wav",
        "comparison": f"{DEMO_PREFIX}/{clip_name}-compare.png",
    }
    metadata = _clip_metadata(clip_name, settings)

    if not force and all(storage.exists(key) for key in keys.values()):
        return Showcase(
            available=True,
            clip_name=clip_name,
            original_audio_key=keys["original"],
            cleaned_audio_key=keys["cleaned"],
            comparison_key=keys["comparison"],
            metrics=_load_cached_metrics(storage, clip_name) or metadata.get("metrics"),
            **{key: metadata[key] for key in ("call_type", "source_file", "noise_source")},
        )

    try:
        audio = load_analysis_audio(settings.call_clips_dir / f"{clip_name}.wav")
        noise_path = noise_reference_path(clip_name, settings)
        noise = load_analysis_audio(noise_path) if noise_path else None
        result = clean_signal(audio, get_profile(metadata["call_type"]), noise_reference=noise)

        playback_rate = settings.playback_sample_rate
        write_audio(
            storage.path_for(keys["original"]),
            resample(audio, settings.sample_rate, playback_rate),
            playback_rate,
        )
        write_audio(
            storage.path_for(keys["cleaned"]),
            resample(result.audio, settings.sample_rate, playback_rate),
            playback_rate,
        )
        render_comparison(audio, result.audio, settings.sample_rate, storage.path_for(keys["comparison"]))
        metrics = {
            "retention": round(result.metrics.target_retention_pct, 1),
            "suppression": round(result.metrics.machine_suppression_pct, 1),
            "gain_db": round(result.metrics.target_machine_gain_db, 1),
            "fidelity": round(result.metrics.spectral_fidelity, 3),
            "preset": result.preset.name,
        }
        _write_cached_metrics(storage, clip_name, metrics)
    except Exception as exc:  # pragma: no cover - showcase must never break the page
        logger.warning("could not build the landing-page showcase: %s", exc)
        return Showcase(available=False)

    return Showcase(
        available=True,
        clip_name=clip_name,
        original_audio_key=keys["original"],
        cleaned_audio_key=keys["cleaned"],
        comparison_key=keys["comparison"],
        metrics=metrics,
        **{key: metadata[key] for key in ("call_type", "source_file", "noise_source")},
    )


def _clip_metadata(clip_name: str, settings: Settings) -> dict[str, Any]:
    try:
        for annotation in load_annotations(settings):
            if annotation.clip_name == clip_name:
                return {
                    "call_type": annotation.call_type,
                    "source_file": annotation.source_file,
                    "noise_source": annotation.noise_source,
                    "metrics": None,
                }
    except FileNotFoundError:
        pass
    return {"call_type": "default", "source_file": "", "noise_source": "unknown", "metrics": None}


def _metrics_path(storage: MediaStorage, clip_name: str) -> Path:
    return storage.path_for(f"{DEMO_PREFIX}/{clip_name}-metrics.json")


def _load_cached_metrics(storage: MediaStorage, clip_name: str) -> dict[str, Any] | None:
    import json

    path = _metrics_path(storage, clip_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_cached_metrics(storage: MediaStorage, clip_name: str, metrics: dict[str, Any]) -> None:
    import json

    path = _metrics_path(storage, clip_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics), encoding="utf-8")
