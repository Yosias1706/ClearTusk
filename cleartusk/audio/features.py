"""Feature extraction for the call-activity detector.

One STFT is reused for every descriptor so a full sliding-window sweep over a
long field recording stays cheap. All features are computed at the analysis
sample rate (2 kHz), where the entire elephant repertoire lives.
"""

from __future__ import annotations

from typing import Final

import librosa
import numpy as np

from cleartusk.audio.presets import Band

#: Sub-bands used to describe the spectral balance of a window.
FEATURE_BANDS: Final[tuple[Band, ...]] = (
    (5.0, 20.0),
    (20.0, 40.0),
    (40.0, 80.0),
    (80.0, 150.0),
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
)
N_MFCC: Final[int] = 13
N_MELS: Final[int] = 24
FRAME_N_FFT: Final[int] = 512
FRAME_HOP: Final[int] = 128
_EPS = 1e-10


def _band_labels() -> list[str]:
    return [f"{int(low)}_{int(high)}hz" for low, high in FEATURE_BANDS]


FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    [f"log_energy_{label}" for label in _band_labels()]
    + [f"energy_share_{label}" for label in _band_labels()]
    + [
        "low_high_ratio_db",
        "spectral_flatness_mean",
        "spectral_flatness_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean",
        "rms_mean",
        "rms_std",
        "rms_crest",
        "rms_dynamic_range_db",
        "zcr_mean",
        "target_band_modulation",
        "harmonic_ridge_mean",
        "peak_frequency_hz",
        "peak_prominence_db",
    ]
    + [f"mfcc{i + 1}_mean" for i in range(N_MFCC)]
    + [f"mfcc{i + 1}_std" for i in range(N_MFCC)]
)
FEATURE_DIM: Final[int] = len(FEATURE_NAMES)


def extract_features(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return a fixed-length descriptor for one audio window."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size < FRAME_N_FFT:
        signal = np.pad(signal, (0, FRAME_N_FFT - signal.size))

    magnitude = np.abs(librosa.stft(signal, n_fft=FRAME_N_FFT, hop_length=FRAME_HOP, win_length=FRAME_N_FFT))
    power = magnitude**2
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=FRAME_N_FFT)
    total_power = float(power.sum()) + _EPS

    log_energies: list[float] = []
    shares: list[float] = []
    band_frames: dict[Band, np.ndarray] = {}
    for band in FEATURE_BANDS:
        mask = (freqs >= band[0]) & (freqs < band[1])
        band_power = power[mask] if np.any(mask) else np.zeros((1, power.shape[1]))
        frame_energy = band_power.sum(axis=0)
        band_frames[band] = frame_energy
        log_energies.append(float(np.log10(frame_energy.sum() + _EPS)))
        shares.append(float(frame_energy.sum() / total_power))

    low_power = sum(band_frames[band].sum() for band in FEATURE_BANDS[:4])
    high_power = sum(band_frames[band].sum() for band in FEATURE_BANDS[4:])
    low_high_ratio_db = float(10 * np.log10((low_power + _EPS) / (high_power + _EPS)))

    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate, roll_percent=0.85)[0]
    rms = librosa.feature.rms(S=magnitude, frame_length=FRAME_N_FFT)[0]
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=FRAME_N_FFT, hop_length=FRAME_HOP)[0]

    rms_mean = float(np.mean(rms))
    rms_peak = float(np.max(rms)) if rms.size else 0.0
    rms_floor = float(np.percentile(rms, 10)) if rms.size else 0.0

    # Energy fluctuation inside the rumble band separates a call (structured
    # amplitude envelope) from steady engine noise (flat envelope).
    target_frames = band_frames[(20.0, 40.0)] + band_frames[(40.0, 80.0)] + band_frames[(80.0, 150.0)]
    target_modulation = float(np.std(target_frames) / (np.mean(target_frames) + _EPS))

    # How strongly bins stand out against their local time neighbourhood.
    smoothed = np.maximum.reduce([np.roll(magnitude, shift, axis=1) for shift in range(-3, 4)])
    harmonic_ridge = float(np.mean(magnitude / (smoothed + _EPS)))

    mean_spectrum = magnitude.mean(axis=1)
    peak_index = int(np.argmax(mean_spectrum))
    peak_frequency = float(freqs[peak_index])
    peak_prominence_db = float(
        20 * np.log10((mean_spectrum[peak_index] + _EPS) / (np.median(mean_spectrum) + _EPS))
    )

    mel = librosa.feature.melspectrogram(
        S=power, sr=sample_rate, n_mels=N_MELS, fmin=5.0, fmax=min(1000.0, sample_rate / 2)
    )
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel + _EPS), n_mfcc=N_MFCC)

    features = np.concatenate(
        [
            np.asarray(log_energies, dtype=np.float64),
            np.asarray(shares, dtype=np.float64),
            np.asarray(
                [
                    low_high_ratio_db,
                    float(np.mean(flatness)),
                    float(np.std(flatness)),
                    float(np.mean(centroid)),
                    float(np.std(centroid)),
                    float(np.mean(bandwidth)),
                    float(np.mean(rolloff)),
                    rms_mean,
                    float(np.std(rms)),
                    float(rms_peak / (rms_mean + _EPS)),
                    float(20 * np.log10((rms_peak + _EPS) / (rms_floor + _EPS))),
                    float(np.mean(zcr)),
                    target_modulation,
                    harmonic_ridge,
                    peak_frequency,
                    peak_prominence_db,
                ],
                dtype=np.float64,
            ),
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def infer_call_type(signal: np.ndarray, sample_rate: int) -> tuple[str, float]:
    """Heuristic call-type guess from where the spectral mass sits.

    Returns the profile name and the share of in-band energy backing the guess.
    This is a deterministic rule, not a learned classifier: the annotated corpus
    is 94 % rumble, which is far too skewed to train an honest type classifier.
    """
    if signal.size == 0:
        return "default", 0.0

    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    power = np.abs(np.fft.rfft(signal)) ** 2
    in_band = (freqs >= 5) & (freqs <= 1000)
    total = float(power[in_band].sum()) + _EPS

    def share(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(power[mask].sum() / total)

    rumble_share = share(10, 150)
    roar_share = share(150, 300)
    trumpet_share = share(300, 800)

    if trumpet_share > 0.30 and trumpet_share > rumble_share:
        return "trumpet", trumpet_share
    if roar_share > 0.25 and roar_share > rumble_share * 0.8:
        return "roar", roar_share
    if rumble_share > 0.45:
        return "rumble", rumble_share
    return "default", max(rumble_share, roar_share, trumpet_share)
