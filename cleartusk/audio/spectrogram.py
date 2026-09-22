"""Spectrogram rendering.

Original and cleaned panels are always drawn on a *shared* decibel scale so the
before/after comparison is honest: any visible difference is signal, not
auto-ranging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import librosa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cleartusk.config import get_settings  # noqa: E402

BACKGROUND = "#14110c"
FOREGROUND = "#f3ece0"
GRID = "#8a7e6b"
ACCENT = "#f0b64b"
COLORMAP = "magma"


@dataclass(frozen=True, slots=True)
class SpectrogramData:
    """Decibel-scaled STFT ready for plotting."""

    values: np.ndarray
    duration_seconds: float
    max_khz: float

    def limits(self, low_percentile: float = 42.0, high_percentile: float = 99.8) -> tuple[float, float]:
        return (
            float(np.percentile(self.values, low_percentile)),
            float(np.percentile(self.values, high_percentile)),
        )


def compute_spectrogram(signal: np.ndarray, sample_rate: int) -> SpectrogramData:
    """STFT magnitude in dB, cropped to the band the app displays."""
    settings = get_settings()
    if signal.size == 0:
        raise ValueError("Cannot render a spectrogram for empty audio.")

    n_fft = min(settings.spectrogram_n_fft, max(256, 2 ** int(np.floor(np.log2(max(signal.size, 256))))))
    hop_length = max(32, n_fft // 8)
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    values = librosa.amplitude_to_db(np.abs(stft) + 1e-9, ref=np.max)
    return SpectrogramData(
        values=values,
        duration_seconds=float(signal.size / sample_rate),
        max_khz=min(settings.spectrogram_max_hz / 1000, sample_rate / 2000),
    )


def render_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    output_path: Path,
    title: str = "Spectrogram",
    events: list[tuple[float, float]] | None = None,
) -> Path:
    """Render a single spectrogram panel, optionally annotating detected events."""
    settings = get_settings()
    data = compute_spectrogram(signal, sample_rate)
    vmin, vmax = data.limits()

    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    image = ax.imshow(
        data.values,
        origin="lower",
        aspect="auto",
        cmap=COLORMAP,
        vmin=vmin,
        vmax=vmax,
        extent=[0, data.duration_seconds, 0, sample_rate / 2000],
        interpolation="bicubic",
    )
    ax.set_ylim(0, data.max_khz)
    ax.set_yticks(np.arange(0, data.max_khz + 0.001, 0.1))
    ax.set_xlabel("Time (s)", color=FOREGROUND, fontsize=9)
    ax.set_ylabel("Frequency (kHz)", color=FOREGROUND, fontsize=9)
    ax.set_title(title, color=FOREGROUND, fontsize=11, loc="left", pad=8)
    ax.tick_params(colors=GRID, labelsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color=GRID, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_alpha(0.4)

    for start, end in events or []:
        ax.axvspan(start, end, color=ACCENT, alpha=0.16)
        ax.axvline(start, color=ACCENT, linewidth=0.8, alpha=0.7)
        ax.axvline(end, color=ACCENT, linewidth=0.8, alpha=0.7)

    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("dB", color=FOREGROUND, fontsize=8)
    colorbar.ax.tick_params(colors=GRID, labelsize=7)
    colorbar.outline.set_edgecolor(GRID)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=settings.spectrogram_dpi, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_comparison(
    original: np.ndarray,
    cleaned: np.ndarray,
    sample_rate: int,
    output_path: Path,
    events: list[tuple[float, float]] | None = None,
) -> Path:
    """Render original above cleaned on one shared colour scale."""
    settings = get_settings()
    original_data = compute_spectrogram(original, sample_rate)
    cleaned_data = compute_spectrogram(cleaned, sample_rate)
    vmin = min(original_data.limits()[0], cleaned_data.limits()[0])
    vmax = max(original_data.limits()[1], cleaned_data.limits()[1])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), facecolor=BACKGROUND, sharex=True)
    image = None
    for ax, data, title in (
        (axes[0], original_data, "Original"),
        (axes[1], cleaned_data, "Cleaned"),
    ):
        ax.set_facecolor(BACKGROUND)
        image = ax.imshow(
            data.values,
            origin="lower",
            aspect="auto",
            cmap=COLORMAP,
            vmin=vmin,
            vmax=vmax,
            extent=[0, data.duration_seconds, 0, sample_rate / 2000],
            interpolation="bicubic",
        )
        ax.set_ylim(0, data.max_khz)
        ax.set_yticks(np.arange(0, data.max_khz + 0.001, 0.1))
        ax.set_ylabel("Frequency (kHz)", color=FOREGROUND, fontsize=9)
        ax.set_title(title, color=FOREGROUND, fontsize=11, loc="left", pad=6)
        ax.tick_params(colors=GRID, labelsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, color=GRID, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_color(GRID)
            spine.set_alpha(0.4)
        for start, end in events or []:
            ax.axvspan(start, end, color=ACCENT, alpha=0.14)

    axes[1].set_xlabel("Time (s)", color=FOREGROUND, fontsize=9)
    colorbar = fig.colorbar(image, ax=axes, pad=0.01)
    colorbar.set_label("dB", color=FOREGROUND, fontsize=8)
    colorbar.ax.tick_params(colors=GRID, labelsize=7)
    colorbar.outline.set_edgecolor(GRID)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=settings.spectrogram_dpi, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_from_file(audio_path: Path, output_path: Path, title: str = "Spectrogram") -> Path:
    """Convenience wrapper: load a file at its native rate and render one panel."""
    from cleartusk.audio.io import load_native_audio

    signal, sample_rate = load_native_audio(audio_path)
    return render_spectrogram(signal, sample_rate, output_path, title=title)
