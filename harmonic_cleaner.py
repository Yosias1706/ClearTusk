from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import ndimage, signal


BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_DIR / "Elephant_Training_Snippets"
NOISE_FOLDER = BASE_DIR / "Elephant_Safe_Noise"
CONTEXT_FOLDER = BASE_DIR / "Elephant_Call_Context"
OUTPUT_FOLDER = BASE_DIR / "Harmonic_Isolated_Clips_1_1000Hz"
REPORT_PATH = BASE_DIR / "metrics_report.csv"
MASTER_CSV = BASE_DIR / "Audio_Files_Master.csv"

SR = 2000
PLAYBACK_SR = 16000
FMIN = 1
FMAX = 1000
CORE_RUMBLE_BAND = (10, 150)
MACHINE_BAND = (150, 1000)
BASE_N_FFT = 2048
BASE_HOP_LENGTH = 128
BASE_WIN_LENGTH = 1024

CALL_TYPE_CONFIGS: dict[str, dict[str, object]] = {
    "rumble": {
        # Main frequency band we want to preserve for this call type.
        "target_band": (10, 150),
        # Extra emphasis given to the target band inside the frequency mask.
        "freq_boost": 1.4,
        # Minimum amount of higher-frequency content allowed through after tapering.
        "upper_taper_floor": 0.40,
        "presets": [
            # HPSS settings separate smoother call structure from rougher/noisier content.
            # threshold / softness control how strict and how gentle the noise gate is.
            # harmonic_blend mixes some harmonic content back in after masking.
            # percussive_reject subtracts more rough/noisy content.
            # target_reinject restores a small amount of original target-band sound.
            {"name": "rumble_preserve", "hpss_margin": (1.1, 2.6), "threshold": 1.08, "softness": 1.85, "harmonic_blend": 0.36, "percussive_reject": 0.02, "target_reinject": 0.26},
            {"name": "rumble_conservative", "hpss_margin": (1.2, 3.0), "threshold": 1.20, "softness": 1.50, "harmonic_blend": 0.28, "percussive_reject": 0.03, "target_reinject": 0.18},
            {"name": "rumble_balanced", "hpss_margin": (1.5, 4.0), "threshold": 1.35, "softness": 1.10, "harmonic_blend": 0.15, "percussive_reject": 0.05, "target_reinject": 0.10},
            {"name": "rumble_aggressive", "hpss_margin": (1.8, 5.0), "threshold": 1.50, "softness": 0.95, "harmonic_blend": 0.10, "percussive_reject": 0.07, "target_reinject": 0.06},
        ],
    },
    "trumpet": {
        "target_band": (80, 450),
        "freq_boost": 1.2,
        "upper_taper_floor": 0.55,
        "presets": [
            {"name": "trumpet_preserve", "hpss_margin": (1.0, 2.3), "threshold": 1.02, "softness": 1.90, "harmonic_blend": 0.40, "percussive_reject": 0.02, "target_reinject": 0.24},
            {"name": "trumpet_conservative", "hpss_margin": (1.1, 2.5), "threshold": 1.10, "softness": 1.70, "harmonic_blend": 0.35, "percussive_reject": 0.02, "target_reinject": 0.18},
            {"name": "trumpet_balanced", "hpss_margin": (1.3, 3.2), "threshold": 1.25, "softness": 1.25, "harmonic_blend": 0.22, "percussive_reject": 0.04, "target_reinject": 0.10},
        ],
    },
    "roar": {
        "target_band": (40, 300),
        "freq_boost": 1.25,
        "upper_taper_floor": 0.50,
        "presets": [
            {"name": "roar_preserve", "hpss_margin": (1.1, 2.7), "threshold": 1.05, "softness": 1.85, "harmonic_blend": 0.36, "percussive_reject": 0.02, "target_reinject": 0.24},
            {"name": "roar_conservative", "hpss_margin": (1.2, 3.0), "threshold": 1.12, "softness": 1.55, "harmonic_blend": 0.30, "percussive_reject": 0.03, "target_reinject": 0.16},
            {"name": "roar_balanced", "hpss_margin": (1.4, 3.8), "threshold": 1.28, "softness": 1.20, "harmonic_blend": 0.18, "percussive_reject": 0.05, "target_reinject": 0.10},
        ],
    },
}
DEFAULT_CALL_CONFIG = {
    "target_band": (15, 220),
    "freq_boost": 1.3,
    "upper_taper_floor": 0.45,
    "presets": [
        {"name": "default_preserve", "hpss_margin": (1.1, 2.6), "threshold": 1.06, "softness": 1.85, "harmonic_blend": 0.36, "percussive_reject": 0.02, "target_reinject": 0.22},
        {"name": "default_conservative", "hpss_margin": (1.2, 3.0), "threshold": 1.18, "softness": 1.55, "harmonic_blend": 0.28, "percussive_reject": 0.03, "target_reinject": 0.16},
        {"name": "default_balanced", "hpss_margin": (1.5, 4.0), "threshold": 1.32, "softness": 1.20, "harmonic_blend": 0.18, "percussive_reject": 0.05, "target_reinject": 0.10},
    ],
}

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def resolve_stft_params(length: int) -> tuple[int, int, int]:
    n_fft = min(BASE_N_FFT, max(256, 2 ** int(np.floor(np.log2(max(length, 256))))))
    win_length = min(BASE_WIN_LENGTH, n_fft)
    hop_length = max(32, win_length // 8)
    return n_fft, hop_length, win_length


def normalize_call_type(raw_call_type: str) -> str:
    text = str(raw_call_type).strip().lower()
    if "trumpet" in text:
        return "trumpet"
    if "roar" in text:
        return "roar"
    if "rumble" in text:
        return "rumble"
    return "default"


def get_call_metadata() -> dict[str, str]:
    if not MASTER_CSV.exists():
        return {}
    df = pd.read_csv(MASTER_CSV, usecols=["Selection", "Call_type"])
    return {
        f"call_{int(row.Selection)}": normalize_call_type(row.Call_type)
        for row in df.itertuples(index=False)
    }


def band_energy(y: np.ndarray, sr: int, band: tuple[float, float]) -> float:
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    power = np.abs(np.fft.rfft(y)) ** 2
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def get_machine_band(target_band: tuple[float, float]) -> tuple[float, float]:
    # Score "machine noise" only above the protected call band so we do not
    # accidentally count preserved elephant sound as noise.
    low = max(MACHINE_BAND[0], target_band[1] + 25.0)
    high = MACHINE_BAND[1]
    if low >= high:
        low = max(MACHINE_BAND[0], high - 50.0)
    return float(low), float(high)


def average_band_spectrum(y: np.ndarray, sr: int, band: tuple[float, float]) -> np.ndarray:
    n_fft, hop_length, win_length = resolve_stft_params(len(y))
    stft = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.zeros(1, dtype=float)
    return np.mean(stft[mask, :], axis=1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    a = a[:length]
    b = b[:length]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def to_db_ratio(numerator: float, denominator: float) -> float:
    eps = 1e-12
    return float(10 * np.log10((numerator + eps) / (denominator + eps)))


def build_frequency_weight(
    freqs: np.ndarray,
    target_band: tuple[float, float],
    freq_boost: float,
    upper_taper_floor: float,
) -> np.ndarray:
    # This mask boosts the target band, tapers higher frequencies instead of
    # cutting them off completely, and rejects content outside our analysis band.
    weight = np.ones_like(freqs)
    elephant_low, elephant_high = target_band

    # Strongly favor the core elephant rumble band.
    core = (freqs >= elephant_low) & (freqs <= elephant_high)
    weight[core] = freq_boost

    # Taper the upper band instead of hard-zeroing it, which helps keep useful harmonics.
    upper = freqs > elephant_high
    weight[upper] = np.clip(
        1.0 - ((freqs[upper] - elephant_high) / (FMAX - elephant_high + 1e-9)) * 0.55,
        upper_taper_floor,
        1.0,
    )

    below = freqs < elephant_low
    weight[below] = 0.65
    weight[freqs > FMAX] = 0.0
    weight[freqs < FMIN] = 0.0
    return weight[:, None]


def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def extract_target_component(y: np.ndarray, target_band: tuple[float, float]) -> np.ndarray:
    nyquist = SR * 0.5
    low = max(target_band[0] / nyquist, 1e-5)
    high = min(target_band[1] / nyquist, 0.95)
    if low >= high:
        return np.zeros_like(y)
    b, a = signal.butter(4, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, y)


def match_target_band_energy(
    candidate: np.ndarray,
    original: np.ndarray,
    target_band: tuple[float, float],
    max_ratio: float = 1.0,
) -> np.ndarray:
    # Cap the cleaned clip so it cannot end up with more target-band sound than
    # the original clip. This keeps the retention metric from being inflated.
    original_energy = band_energy(original, SR, target_band)
    candidate_energy = band_energy(candidate, SR, target_band)
    if candidate_energy <= 0 or original_energy <= 0:
        return candidate

    allowed_energy = original_energy * max_ratio
    if candidate_energy <= allowed_energy:
        return candidate

    adjusted = candidate.astype(np.float32)
    for _ in range(4):
        candidate_energy = band_energy(adjusted, SR, target_band)
        if candidate_energy <= allowed_energy * 1.0001:
            break
        scale = np.sqrt(allowed_energy / candidate_energy)
        target_component = extract_target_component(adjusted, target_band)
        adjusted = adjusted - target_component + (target_component * scale)
    return adjusted.astype(np.float32)


def isolate_elephant_bands(
    y: np.ndarray,
    noise_reference: np.ndarray | None,
    target_band: tuple[float, float],
    preset: dict[str, float | tuple[float, float] | str],
    freq_boost: float,
    upper_taper_floor: float,
) -> np.ndarray:
    # Read the preset knobs that define how cautious or aggressive this pass is.
    hpss_margin = preset["hpss_margin"]
    noise_gate_threshold = float(preset["threshold"])
    noise_gate_softness = float(preset["softness"])
    harmonic_blend = float(preset["harmonic_blend"])
    percussive_reject = float(preset["percussive_reject"])
    target_reinject = float(preset.get("target_reinject", 0.0))

    # Separate smoother harmonic content from rougher/percussive content.
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=hpss_margin)
    original_target_component = extract_target_component(y, target_band)
    n_fft, hop_length, win_length = resolve_stft_params(len(y_harmonic))

    stft = librosa.stft(y_harmonic, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude, phase = librosa.magphase(stft)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)

    # Build a harmonic-friendly mask so sustained elephant structure is favored.
    harmonic_peak = ndimage.maximum_filter(magnitude, size=(1, 11))
    harmonic_mask = np.clip(magnitude / (harmonic_peak + 1e-9), 0.0, 1.0)
    harmonic_mask = np.clip((harmonic_mask - 0.18) / 0.82, 0.0, 1.0)

    if noise_reference is not None and len(noise_reference) > 0:
        # If we have a safe-noise clip from the same recording, estimate a local
        # noise profile and gate the signal against it.
        noise_n_fft, noise_hop_length, noise_win_length = resolve_stft_params(len(noise_reference))
        noise_stft = np.abs(
            librosa.stft(
                noise_reference,
                n_fft=noise_n_fft,
                hop_length=noise_hop_length,
                win_length=noise_win_length,
            )
        )
        if noise_stft.shape[0] != magnitude.shape[0]:
            noise_stft = librosa.util.fix_length(noise_stft, size=magnitude.shape[0], axis=0)
        noise_profile = np.median(noise_stft, axis=1, keepdims=True)
        snr_like = magnitude / (noise_profile + 1e-9)
        noise_gate = np.clip(
            (snr_like - noise_gate_threshold) / noise_gate_softness,
            0.0,
            1.0,
        )
    else:
        noise_gate = np.ones_like(magnitude)

    freq_weight = build_frequency_weight(freqs, target_band, freq_boost, upper_taper_floor)
    combined_mask = harmonic_mask * noise_gate * freq_weight
    combined_mask = ndimage.gaussian_filter(combined_mask, sigma=(1.0, 1.0))
    combined_mask = np.clip(combined_mask, 0.0, 1.0)

    # Apply the mask and reconstruct the time-domain signal.
    cleaned_stft = magnitude * combined_mask * phase
    y_clean = librosa.istft(cleaned_stft, hop_length=hop_length, win_length=win_length, length=len(y))

    # Blend a little harmonic and target-band content back in so the elephant
    # call survives aggressive denoising.
    base_mix = max(0.55, 1.0 - harmonic_blend + percussive_reject)
    y_clean = base_mix * y_clean + harmonic_blend * y_harmonic - percussive_reject * y_percussive
    y_clean = y_clean + target_reinject * original_target_component

    # Final low-frequency emphasis with a gentle low-pass.
    normalized_cutoff = min(FMAX / (SR * 0.5), 0.95)
    b, a = signal.butter(4, normalized_cutoff, btype="lowpass")
    y_clean = signal.filtfilt(b, a, y_clean)
    y_clean = match_target_band_energy(y_clean, y, target_band, max_ratio=0.999)

    peak = np.max(np.abs(y_clean))
    if peak > 0:
        y_clean = y_clean / peak * min(peak, 0.98)

    return y_clean.astype(np.float32)


def calculate_metrics(
    original: np.ndarray,
    cleaned: np.ndarray,
    noise_reference: np.ndarray | None,
    context: np.ndarray | None,
    target_band: tuple[float, float],
) -> dict[str, float | int | str]:
    # These metrics are proxies:
    # retention = how much elephant-band sound stayed,
    # suppression = how much upper-band machine sound dropped,
    # ratio gain = whether the elephant stands out more after cleaning,
    # similarity = whether the target-band sound profile still resembles the original.
    machine_band = get_machine_band(target_band)
    rms_orig = float(np.sqrt(np.mean(original**2)))
    rms_clean = float(np.sqrt(np.mean(cleaned**2)))
    reduction_pct = (1 - (rms_clean / rms_orig)) * 100 if rms_orig > 0 else 0.0

    n_fft, hop_length, win_length = resolve_stft_params(len(cleaned))
    stft_clean = np.abs(librosa.stft(cleaned, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    sparsity_pct = float((stft_clean < 0.01 * np.max(stft_clean + 1e-9)).sum() / stft_clean.size * 100)

    freqs = np.fft.rfftfreq(len(cleaned), 1 / SR)
    fft_data = np.abs(np.fft.rfft(cleaned))
    peak_freq_hz = float(freqs[np.argmax(fft_data)]) if len(fft_data) else 0.0
    peak_in_target_band = int(target_band[0] <= peak_freq_hz <= target_band[1])

    elephant_orig = band_energy(original, SR, target_band)
    elephant_clean = band_energy(cleaned, SR, target_band)
    machine_orig = band_energy(original, SR, machine_band)
    machine_clean = band_energy(cleaned, SR, machine_band)
    rumble_orig = band_energy(original, SR, CORE_RUMBLE_BAND)
    rumble_clean = band_energy(cleaned, SR, CORE_RUMBLE_BAND)

    elephant_retention_pct = min((elephant_clean / (elephant_orig + 1e-12)) * 100, 100.0)
    machine_suppression_pct = (1 - (machine_clean / (machine_orig + 1e-12))) * 100
    emr_gain_db = to_db_ratio(elephant_clean, machine_clean) - to_db_ratio(elephant_orig, machine_orig)
    rumble_retention_pct = (rumble_clean / (rumble_orig + 1e-12)) * 100

    elephant_similarity = cosine_similarity(
        average_band_spectrum(original, SR, target_band),
        average_band_spectrum(cleaned, SR, target_band),
    )

    metrics: dict[str, float | int | str] = {
        "reduction_pct": reduction_pct,
        "sparsity_pct": sparsity_pct,
        "peak_freq_hz": peak_freq_hz,
        "peak_in_target_band": peak_in_target_band,
        "target_band_retention_pct": float(elephant_retention_pct),
        "machine_band_suppression_pct": float(machine_suppression_pct),
        "target_machine_ratio_gain_db": float(emr_gain_db),
        "target_band_cosine_similarity": float(elephant_similarity),
        "core_rumble_retention_pct": float(rumble_retention_pct),
        "target_band_low_hz": float(target_band[0]),
        "target_band_high_hz": float(target_band[1]),
        "machine_band_low_hz": float(machine_band[0]),
        "machine_band_high_hz": float(machine_band[1]),
    }

    if noise_reference is not None and len(noise_reference) > 0:
        noise_orig = band_energy(original, SR, machine_band)
        noise_clean = band_energy(cleaned, SR, machine_band)
        noise_ref_energy = band_energy(noise_reference, SR, machine_band)
        metrics["noise_reference_match_reduction_db"] = (
            to_db_ratio(noise_orig, noise_ref_energy) - to_db_ratio(noise_clean, noise_ref_energy)
        )
    else:
        metrics["noise_reference_match_reduction_db"] = np.nan

    if context is not None and len(context) > 0:
        context_elephant = band_energy(context, SR, CORE_RUMBLE_BAND)
        metrics["context_elephant_band_energy"] = float(context_elephant)
    else:
        metrics["context_elephant_band_energy"] = np.nan

    return metrics


def score_candidate(metrics: dict[str, float | int | str], call_type: str) -> float:
    # Combine the main preservation and denoising metrics into one score so we
    # can choose the best preset for each clip automatically.
    target_retention = float(metrics["target_band_retention_pct"])
    suppression = float(metrics["machine_band_suppression_pct"])
    ratio_gain = float(metrics["target_machine_ratio_gain_db"])
    similarity = float(metrics["target_band_cosine_similarity"])
    peak_bonus = 3.0 if int(metrics["peak_in_target_band"]) else -2.0

    retention_floor = 40.0 if call_type == "rumble" else 30.0
    retention_penalty = max(0.0, retention_floor - target_retention) * 0.75
    suppression_cap = min(suppression, 95.0)

    return (
        ratio_gain * 1.0
        + suppression_cap * 0.04
        + target_retention * 0.22
        + similarity * 14.0
        + peak_bonus
        - retention_penalty
    )


def main() -> None:
    # Batch-process every training clip, try all presets for that clip's call
    # type, keep the highest-scoring result, and write a full report.
    records: list[dict[str, float | int | str]] = []
    call_type_lookup = get_call_metadata()

    print(f"Processing clips in {INPUT_FOLDER.name} with local noise references when available...")

    for clip_path in sorted(INPUT_FOLDER.glob("*.wav")):
        clip_id = clip_path.stem
        call_id = clip_id.replace("call_", "")
        call_type = call_type_lookup.get(clip_id, "default")
        call_config = CALL_TYPE_CONFIGS.get(call_type, DEFAULT_CALL_CONFIG)
        target_band = call_config["target_band"]
        noise_path = NOISE_FOLDER / f"call_{call_id}_noise.wav"
        context_path = CONTEXT_FOLDER / f"call_{call_id}_context.wav"

        original = load_audio(clip_path)
        noise_reference = load_audio(noise_path) if noise_path.exists() else None
        context = load_audio(context_path) if context_path.exists() else None

        best_result: tuple[np.ndarray, dict[str, float | int | str], str, float] | None = None
        for preset in call_config["presets"]:
            cleaned_candidate = isolate_elephant_bands(
                original,
                noise_reference=noise_reference,
                target_band=target_band,
                preset=preset,
                freq_boost=float(call_config["freq_boost"]),
                upper_taper_floor=float(call_config["upper_taper_floor"]),
            )
            metrics_candidate = calculate_metrics(
                original,
                cleaned_candidate,
                noise_reference,
                context,
                target_band=target_band,
            )
            candidate_score = score_candidate(metrics_candidate, call_type)
            if best_result is None or candidate_score > best_result[3]:
                best_result = (
                    cleaned_candidate,
                    metrics_candidate,
                    str(preset["name"]),
                    candidate_score,
                )

        assert best_result is not None
        cleaned, metrics, best_preset_name, best_score = best_result

        output_path = OUTPUT_FOLDER / clip_path.name
        cleaned_for_playback = librosa.resample(cleaned, orig_sr=SR, target_sr=PLAYBACK_SR)
        sf.write(output_path, cleaned_for_playback, PLAYBACK_SR, subtype="PCM_16")

        record = {
            "clip": clip_path.name,
            "call_type": call_type,
            "selected_preset": best_preset_name,
            "candidate_score": best_score,
            "has_safe_noise": int(noise_reference is not None),
            "has_context": int(context is not None),
            **metrics,
        }
        records.append(record)

        print(
            f"{clip_path.name} | "
            f"type={call_type} | "
            f"preset={best_preset_name} | "
            f"retain={metrics['target_band_retention_pct']:.1f}% | "
            f"suppress={metrics['machine_band_suppression_pct']:.1f}% | "
            f"ratio_gain={metrics['target_machine_ratio_gain_db']:.2f} dB"
        )

    report = pd.DataFrame(records).sort_values(
        by=[
            "target_machine_ratio_gain_db",
            "target_band_retention_pct",
            "machine_band_suppression_pct",
        ],
        ascending=[False, False, False],
    )
    summary_lines = [
        "summary_metric,value",
        f"total_files_processed,{len(report)}",
        f"avg_noise_reduction_pct,{report['reduction_pct'].mean():.2f}",
        f"avg_spectral_sparsity_pct,{report['sparsity_pct'].mean():.2f}",
        f"peak_in_target_band_pct,{report['peak_in_target_band'].mean() * 100:.1f}",
        f"avg_target_band_retention_pct,{report['target_band_retention_pct'].mean():.2f}",
        f"avg_machine_band_suppression_pct,{report['machine_band_suppression_pct'].mean():.2f}",
        f"avg_target_machine_gain_db,{report['target_machine_ratio_gain_db'].mean():.2f}",
        f"avg_target_band_similarity,{report['target_band_cosine_similarity'].mean():.3f}",
        f"avg_core_rumble_retention_pct,{report['core_rumble_retention_pct'].mean():.2f}",
        "",
        "detailed_clip_metrics",
    ]
    report_csv = report.to_csv(index=False)
    REPORT_PATH.write_text("\n".join(summary_lines) + "\n" + report_csv, encoding="utf-8")

    print("\n==============================")
    print("FINAL DATA REPORT")
    print("==============================")
    print(f"Total Files Processed: {len(report)}")
    if len(report):
        print(f"Avg Noise Reduction:           {report['reduction_pct'].mean():.2f}%")
        print(f"Avg Spectral Sparsity:         {report['sparsity_pct'].mean():.2f}%")
        print(f"Peak In Target Band:           {report['peak_in_target_band'].mean() * 100:.1f}%")
        print(f"Avg Target-Band Retention:     {report['target_band_retention_pct'].mean():.2f}%")
        print(f"Avg Machine Suppression:       {report['machine_band_suppression_pct'].mean():.2f}%")
        print(f"Avg Target/Machine Gain:       {report['target_machine_ratio_gain_db'].mean():.2f} dB")
        print(f"Avg Target-Band Similarity:    {report['target_band_cosine_similarity'].mean():.3f}")
        print(f"Avg Core Rumble Retention:     {report['core_rumble_retention_pct'].mean():.2f}%")
    print(f"Detailed report saved to:      {REPORT_PATH}")
    print("==============================")


if __name__ == "__main__":
    main()
