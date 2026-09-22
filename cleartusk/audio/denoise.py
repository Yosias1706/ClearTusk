"""The harmonic isolation denoiser.

Pipeline for one candidate pass:

1. Harmonic/percussive source separation splits sustained call structure from
   transient machine clatter.
2. A harmonic-ridge mask keeps bins that stand out along the time axis.
3. An optional noise gate is derived from a "safe noise" reference recorded
   next to the call in the same file.
4. A frequency weight boosts the protected band and tapers (never hard-cuts)
   everything above it.
5. Harmonic content and a slice of the original in-band call are mixed back so
   aggressive masking cannot hollow out the vocalisation.
6. Target-band energy is capped at the original so retention cannot be inflated.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy import ndimage
from scipy import signal as scipy_signal

from cleartusk.audio.io import stft_params
from cleartusk.audio.metrics import QualityMetrics, band_energy, compute_metrics, score_candidate
from cleartusk.audio.presets import Band, CallProfile, Preset
from cleartusk.config import get_settings


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """Best candidate produced for a clip."""

    audio: np.ndarray
    metrics: QualityMetrics
    preset: Preset
    score: float
    profile: CallProfile
    candidates_evaluated: int


def build_frequency_weight(
    freqs: np.ndarray,
    profile: CallProfile,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Per-bin gain: boost the protected band, taper above it, reject outside the analysis band."""
    weight = np.ones_like(freqs)
    low, high = profile.target_band

    weight[(freqs >= low) & (freqs <= high)] = profile.freq_boost

    upper = freqs > high
    weight[upper] = np.clip(
        1.0 - ((freqs[upper] - high) / (fmax - high + 1e-9)) * 0.55,
        profile.upper_taper_floor,
        1.0,
    )
    weight[freqs < low] = 0.65
    weight[freqs > fmax] = 0.0
    weight[freqs < fmin] = 0.0
    return weight[:, None]


def bandpass(signal_data: np.ndarray, band: Band, sample_rate: int) -> np.ndarray:
    """Zero-phase Butterworth bandpass restricted to a safe normalised range."""
    nyquist = sample_rate * 0.5
    low = max(band[0] / nyquist, 1e-5)
    high = min(band[1] / nyquist, 0.95)
    if low >= high or signal_data.size < 24:
        return np.zeros_like(signal_data)
    b, a = scipy_signal.butter(4, [low, high], btype="bandpass")
    return scipy_signal.filtfilt(b, a, signal_data)


def cap_target_band_energy(
    candidate: np.ndarray,
    original: np.ndarray,
    band: Band,
    sample_rate: int,
    max_ratio: float = 0.999,
) -> np.ndarray:
    """Scale the in-band component down if re-injection pushed it past the original."""
    original_energy = band_energy(original, sample_rate, band)
    candidate_energy = band_energy(candidate, sample_rate, band)
    if candidate_energy <= 0 or original_energy <= 0:
        return candidate

    allowed = original_energy * max_ratio
    adjusted = candidate.astype(np.float32)
    for _ in range(4):
        candidate_energy = band_energy(adjusted, sample_rate, band)
        if candidate_energy <= allowed * 1.0001:
            break
        scale = float(np.sqrt(allowed / candidate_energy))
        component = bandpass(adjusted, band, sample_rate)
        adjusted = adjusted - component + component * scale
    return adjusted.astype(np.float32)


def _noise_gate(
    magnitude: np.ndarray,
    noise_reference: np.ndarray | None,
    preset: Preset,
) -> np.ndarray:
    if noise_reference is None or noise_reference.size == 0:
        return np.ones_like(magnitude)

    n_fft, hop_length, win_length = stft_params(len(noise_reference))
    noise_stft = np.abs(
        librosa.stft(noise_reference, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    if noise_stft.shape[0] != magnitude.shape[0]:
        noise_stft = librosa.util.fix_length(noise_stft, size=magnitude.shape[0], axis=0)
    noise_profile = np.median(noise_stft, axis=1, keepdims=True)
    snr_like = magnitude / (noise_profile + 1e-9)
    return np.clip((snr_like - preset.threshold) / preset.softness, 0.0, 1.0)


def isolate_call(
    audio: np.ndarray,
    profile: CallProfile,
    preset: Preset,
    noise_reference: np.ndarray | None = None,
    sample_rate: int | None = None,
) -> np.ndarray:
    """Run a single denoising pass and return the cleaned signal."""
    settings = get_settings()
    sample_rate = sample_rate or settings.sample_rate
    if audio.size == 0:
        raise ValueError("Cannot clean an empty signal.")

    harmonic, percussive = librosa.effects.hpss(audio, margin=preset.hpss_margin)
    original_in_band = bandpass(audio, profile.target_band, sample_rate)

    n_fft, hop_length, win_length = stft_params(len(harmonic))
    stft = librosa.stft(harmonic, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude, phase = librosa.magphase(stft)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    # Keep bins that rise above their local time neighbourhood: sustained
    # harmonic ridges survive, diffuse broadband noise does not.
    ridge = ndimage.maximum_filter(magnitude, size=(1, 11))
    harmonic_mask = np.clip(magnitude / (ridge + 1e-9), 0.0, 1.0)
    harmonic_mask = np.clip((harmonic_mask - 0.18) / 0.82, 0.0, 1.0)

    mask = harmonic_mask * _noise_gate(magnitude, noise_reference, preset)
    mask *= build_frequency_weight(freqs, profile, settings.analysis_fmin_hz, settings.analysis_fmax_hz)
    mask = np.clip(ndimage.gaussian_filter(mask, sigma=(1.0, 1.0)), 0.0, 1.0)

    cleaned = librosa.istft(
        magnitude * mask * phase, hop_length=hop_length, win_length=win_length, length=len(audio)
    )

    base_mix = max(0.55, 1.0 - preset.harmonic_blend + preset.percussive_reject)
    cleaned = base_mix * cleaned + preset.harmonic_blend * harmonic - preset.percussive_reject * percussive
    cleaned = cleaned + preset.target_reinject * original_in_band

    cutoff = min(settings.analysis_fmax_hz / (sample_rate * 0.5), 0.95)
    b, a = scipy_signal.butter(4, cutoff, btype="lowpass")
    cleaned = scipy_signal.filtfilt(b, a, cleaned)
    cleaned = cap_target_band_energy(cleaned, audio, profile.target_band, sample_rate)

    peak = float(np.max(np.abs(cleaned)))
    if peak > 0:
        cleaned = cleaned / peak * min(peak, 0.98)
    return np.asarray(cleaned, dtype=np.float32)


def clean_signal(
    audio: np.ndarray,
    profile: CallProfile,
    noise_reference: np.ndarray | None = None,
    sample_rate: int | None = None,
) -> CleaningResult:
    """Evaluate every preset in the profile and return the highest scoring candidate."""
    sample_rate = sample_rate or get_settings().sample_rate
    best: CleaningResult | None = None

    for preset in profile.presets:
        candidate = isolate_call(audio, profile, preset, noise_reference, sample_rate)
        metrics = compute_metrics(audio, candidate, profile, sample_rate)
        score = score_candidate(metrics, profile)
        if best is None or score > best.score:
            best = CleaningResult(
                audio=candidate,
                metrics=metrics,
                preset=preset,
                score=score,
                profile=profile,
                candidates_evaluated=0,
            )

    if best is None:
        raise ValueError(f"Call profile '{profile.name}' defines no presets.")
    return CleaningResult(
        audio=best.audio,
        metrics=best.metrics,
        preset=best.preset,
        score=best.score,
        profile=best.profile,
        candidates_evaluated=len(profile.presets),
    )
