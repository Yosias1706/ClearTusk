"""Audio loading and writing helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from cleartusk.config import get_settings


@dataclass(frozen=True, slots=True)
class AudioInfo:
    """Lightweight probe result for a media file."""

    duration_seconds: float
    sample_rate: int
    channels: int
    size_bytes: int


def analysis_sample_rate() -> int:
    return get_settings().sample_rate


def load_analysis_audio(path: Path, sample_rate: int | None = None) -> np.ndarray:
    """Load a file as mono float32 at the analysis sample rate."""
    signal, _ = librosa.load(path, sr=sample_rate or analysis_sample_rate(), mono=True)
    return np.asarray(signal, dtype=np.float32)


def load_native_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a file as mono float32 at its native sample rate."""
    signal, sample_rate = librosa.load(path, sr=None, mono=True)
    return np.asarray(signal, dtype=np.float32), int(sample_rate)


def probe(path: Path) -> AudioInfo:
    """Read duration/sample-rate metadata without decoding the whole file when possible."""
    try:
        info = sf.info(str(path))
        return AudioInfo(
            duration_seconds=float(info.duration),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            size_bytes=path.stat().st_size,
        )
    except Exception:
        # Compressed formats soundfile cannot open are decoded through audioread.
        signal, sample_rate = load_native_audio(path)
        return AudioInfo(
            duration_seconds=float(len(signal) / sample_rate) if sample_rate else 0.0,
            sample_rate=int(sample_rate),
            channels=1,
            size_bytes=path.stat().st_size,
        )


def write_audio(path: Path, signal: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> Path:
    """Write a mono signal to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(signal, dtype=np.float32), int(sample_rate), subtype=subtype)
    return path


def resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample only when the rates actually differ."""
    if source_rate == target_rate:
        return signal
    return librosa.resample(signal, orig_sr=source_rate, target_sr=target_rate)


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, used to de-duplicate uploads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stft_params(length: int, base_n_fft: int = 2048, base_win_length: int = 1024) -> tuple[int, int, int]:
    """Pick ``(n_fft, hop_length, win_length)`` that fit a signal of ``length`` samples."""
    n_fft = min(base_n_fft, max(256, 2 ** int(np.floor(np.log2(max(length, 256))))))
    win_length = min(base_win_length, n_fft)
    hop_length = max(32, win_length // 8)
    return n_fft, hop_length, win_length
