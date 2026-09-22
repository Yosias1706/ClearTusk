"""Objective quality metrics for a cleaning pass.

All metrics are *reference-free proxies*: there is no clean ground truth for a
field recording, so quality is expressed as how much protected-band energy
survived versus how much out-of-band machine energy was removed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np

from cleartusk.audio.io import stft_params
from cleartusk.audio.presets import CORE_RUMBLE_BAND, Band, CallProfile

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Outcome of one cleaning candidate."""

    target_retention_pct: float
    machine_suppression_pct: float
    target_machine_gain_db: float
    spectral_fidelity: float
    core_rumble_retention_pct: float
    broadband_reduction_pct: float
    spectral_sparsity_pct: float
    peak_frequency_hz: float
    peak_in_target_band: bool
    target_band_low_hz: float
    target_band_high_hz: float
    machine_band_low_hz: float
    machine_band_high_hz: float

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def band_energy(signal: np.ndarray, sample_rate: int, band: Band) -> float:
    """Total spectral power inside ``band``."""
    if signal.size == 0:
        return 0.0
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.sum(power[mask])) if np.any(mask) else 0.0


def average_band_spectrum(signal: np.ndarray, sample_rate: int, band: Band) -> np.ndarray:
    """Time-averaged magnitude spectrum restricted to ``band``."""
    n_fft, hop_length, win_length = stft_params(len(signal))
    magnitude = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.zeros(1, dtype=float)
    return np.mean(magnitude[mask, :], axis=1)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    left, right = left[:length], right[:length]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def db_ratio(numerator: float, denominator: float) -> float:
    return float(10 * np.log10((numerator + _EPS) / (denominator + _EPS)))


def compute_metrics(
    original: np.ndarray,
    cleaned: np.ndarray,
    profile: CallProfile,
    sample_rate: int,
) -> QualityMetrics:
    """Compare ``cleaned`` against ``original`` under a call profile."""
    target_band = profile.target_band
    machine_band = profile.machine_band

    rms_original = float(np.sqrt(np.mean(original**2))) if original.size else 0.0
    rms_cleaned = float(np.sqrt(np.mean(cleaned**2))) if cleaned.size else 0.0
    broadband_reduction = (1 - rms_cleaned / rms_original) * 100 if rms_original > 0 else 0.0

    n_fft, hop_length, win_length = stft_params(len(cleaned))
    cleaned_stft = np.abs(librosa.stft(cleaned, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    sparsity = float((cleaned_stft < 0.01 * np.max(cleaned_stft + _EPS)).sum() / cleaned_stft.size * 100)

    freqs = np.fft.rfftfreq(len(cleaned), 1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(cleaned))
    peak_frequency = float(freqs[int(np.argmax(spectrum))]) if spectrum.size else 0.0

    target_original = band_energy(original, sample_rate, target_band)
    target_cleaned = band_energy(cleaned, sample_rate, target_band)
    machine_original = band_energy(original, sample_rate, machine_band)
    machine_cleaned = band_energy(cleaned, sample_rate, machine_band)
    rumble_original = band_energy(original, sample_rate, CORE_RUMBLE_BAND)
    rumble_cleaned = band_energy(cleaned, sample_rate, CORE_RUMBLE_BAND)

    gain_db = db_ratio(target_cleaned, machine_cleaned) - db_ratio(target_original, machine_original)

    return QualityMetrics(
        target_retention_pct=float(min(target_cleaned / (target_original + _EPS) * 100, 100.0)),
        machine_suppression_pct=float((1 - machine_cleaned / (machine_original + _EPS)) * 100),
        target_machine_gain_db=float(gain_db),
        spectral_fidelity=cosine_similarity(
            average_band_spectrum(original, sample_rate, target_band),
            average_band_spectrum(cleaned, sample_rate, target_band),
        ),
        core_rumble_retention_pct=float(rumble_cleaned / (rumble_original + _EPS) * 100),
        broadband_reduction_pct=float(broadband_reduction),
        spectral_sparsity_pct=sparsity,
        peak_frequency_hz=peak_frequency,
        peak_in_target_band=bool(target_band[0] <= peak_frequency <= target_band[1]),
        target_band_low_hz=float(target_band[0]),
        target_band_high_hz=float(target_band[1]),
        machine_band_low_hz=float(machine_band[0]),
        machine_band_high_hz=float(machine_band[1]),
    )


def score_candidate(metrics: QualityMetrics, profile: CallProfile) -> float:
    """Collapse the metric vector into one comparable score.

    Weights favour a call that stands out more (``gain_db``) without letting the
    denoiser reach that number by erasing the call itself: retention below the
    profile floor is penalised, and suppression saturates at 95 %.
    """
    peak_bonus = 3.0 if metrics.peak_in_target_band else -2.0
    retention_penalty = max(0.0, profile.retention_floor - metrics.target_retention_pct) * 0.75
    suppression = min(metrics.machine_suppression_pct, 95.0)

    return (
        metrics.target_machine_gain_db
        + suppression * 0.04
        + metrics.target_retention_pct * 0.22
        + metrics.spectral_fidelity * 14.0
        + peak_bonus
        - retention_penalty
    )
