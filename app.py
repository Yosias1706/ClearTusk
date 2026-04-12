from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import librosa  # noqa: F401
import matplotlib
import numpy as np  # noqa: F401
import pandas as pd
import soundfile as sf
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, url_for
from scipy import signal  # noqa: F401

from harmonic_cleaner import (
    CALL_TYPE_CONFIGS,
    DEFAULT_CALL_CONFIG,
    SR,
    calculate_metrics,
    isolate_elephant_bands,
    load_audio,
    score_candidate,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
AUDIO_FOLDER = STATIC_DIR / "audio"
SPECTROGRAM_FOLDER = STATIC_DIR / "spectrograms"
METADATA_FILE = BASE_DIR / "metadata.csv"
ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SPECTROGRAM_MAX_HZ = 700
SPECTROGRAM_N_FFT = 8192

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
app.config["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)
SPECTROGRAM_FOLDER.mkdir(parents=True, exist_ok=True)


def apply_bandpass_filter(
    audio_signal: np.ndarray,
    sample_rate: int,
    lowcut: float = 300.0,
    highcut: float = 3400.0,
) -> np.ndarray:
    """Placeholder bandpass filter for future DSP logic."""
    if audio_signal.size == 0 or sample_rate <= 0:
        return audio_signal

    nyquist = sample_rate * 0.5
    low = max(lowcut / nyquist, 1e-6)
    high = min(highcut / nyquist, 0.999)

    if low >= high:
        return audio_signal

    b, a = signal.butter(N=4, Wn=[low, high], btype="bandpass")
    return signal.filtfilt(b, a, audio_signal)


def load_metadata() -> list[dict]:
    if not METADATA_FILE.exists():
        return []

    df = pd.read_csv(METADATA_FILE)
    records = df.fillna("").to_dict(orient="records")

    for row in records:
        raw_filename = str(row.get("raw_file", "")).strip()
        processed_filename = str(row.get("processed_file", "")).strip()
        row["raw_url"] = url_for("static", filename=f"audio/{raw_filename}") if raw_filename else ""
        row["processed_url"] = (
            url_for("static", filename=f"uploads/{processed_filename}") if processed_filename else ""
        )

    return records


def get_call_config(call_type: str) -> tuple[str, dict[str, object]]:
    normalized = str(call_type).strip().lower() or "default"
    if normalized not in CALL_TYPE_CONFIGS:
        normalized = "default"
    return normalized, CALL_TYPE_CONFIGS.get(normalized, DEFAULT_CALL_CONFIG)


def save_spectrogram(audio_path: Path, prefix: str) -> str:
    signal_data, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    if signal_data.size == 0:
        raise ValueError("Uploaded audio is empty.")

    n_fft = SPECTROGRAM_N_FFT
    hop_length = max(64, n_fft // 8)
    stft = librosa.stft(signal_data, n_fft=n_fft, hop_length=hop_length)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft) + 1e-9, ref=np.max)
    max_display_khz = min(SPECTROGRAM_MAX_HZ / 1000, sample_rate / 2000)
    display_ticks = np.arange(0, max_display_khz + 0.001, 0.1)
    vmin = float(np.percentile(spectrogram_db, 42))
    vmax = float(np.percentile(spectrogram_db, 99.8))

    output_name = f"{prefix}_{uuid4().hex[:8]}.png"
    output_path = SPECTROGRAM_FOLDER / output_name

    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(
        spectrogram_db,
        origin="lower",
        aspect="auto",
        cmap="gray_r",
        vmin=vmin,
        vmax=vmax,
        extent=[0, len(signal_data) / sample_rate, 0, sample_rate / 2000],
        interpolation="bicubic",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_ylim(0, max_display_khz)
    ax.set_yticks(display_ticks)
    ax.grid(axis="y", linestyle=":", linewidth=1.0, color="black", alpha=0.7)
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return url_for("static", filename=f"spectrograms/{output_name}")


def clean_uploaded_clip(input_path: Path, call_type: str) -> tuple[str, dict[str, float | int | str], str]:
    normalized_call_type, call_config = get_call_config(call_type)
    _, original_sample_rate = librosa.load(input_path, sr=None, mono=True)
    original = load_audio(input_path)

    best_result: tuple[np.ndarray, dict[str, float | int | str], str, float] | None = None
    for preset in call_config["presets"]:
        cleaned_candidate = isolate_elephant_bands(
            original,
            noise_reference=None,
            target_band=tuple(call_config["target_band"]),
            preset=preset,
            freq_boost=float(call_config["freq_boost"]),
            upper_taper_floor=float(call_config["upper_taper_floor"]),
        )
        metrics_candidate = calculate_metrics(
            original,
            cleaned_candidate,
            noise_reference=None,
            context=None,
            target_band=tuple(call_config["target_band"]),
        )
        candidate_score = score_candidate(metrics_candidate, normalized_call_type)
        if best_result is None or candidate_score > best_result[3]:
            best_result = (
                cleaned_candidate,
                metrics_candidate,
                str(preset["name"]),
                candidate_score,
            )

    if best_result is None:
        raise ValueError("No valid cleaning preset was produced.")

    cleaned_audio, metrics, selected_preset, _ = best_result
    cleaned_for_export = cleaned_audio
    if original_sample_rate != SR:
        cleaned_for_export = librosa.resample(cleaned_audio, orig_sr=SR, target_sr=original_sample_rate)

    output_name = f"{input_path.stem}_cleaned_{uuid4().hex[:8]}.wav"
    output_path = UPLOAD_FOLDER / output_name
    sf.write(output_path, cleaned_for_export, original_sample_rate, subtype="PCM_16")
    return output_name, metrics, selected_preset


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def index():
    metadata = load_metadata()
    return render_template("index.html", metadata=metadata, call_types=["rumble", "trumpet", "roar", "default"])


@app.route("/process", methods=["POST"])
def process_audio():
    payload = request.get_json(silent=True) or {}
    filename = str(payload.get("filename", "")).strip()

    if not filename:
        return jsonify({"success": False, "error": "No audio file specified."}), 400

    source_path = AUDIO_FOLDER / filename
    if not source_path.exists():
        return jsonify({"success": False, "error": f"Audio file '{filename}' was not found."}), 404

    try:
        audio_signal, sample_rate = librosa.load(source_path, sr=None, mono=True)
        filtered_signal = apply_bandpass_filter(audio_signal, sample_rate)

        output_name = f"{source_path.stem}_processed_{uuid4().hex[:8]}.wav"
        output_path = UPLOAD_FOLDER / output_name
        sf.write(output_path, filtered_signal, sample_rate)

        return jsonify(
            {
                "success": True,
                "message": "Audio processed successfully.",
                "processed_file": output_name,
                "processed_url": url_for("static", filename=f"uploads/{output_name}"),
            }
        )
    except Exception as exc:  # pragma: no cover
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/process-upload", methods=["POST"])
def process_uploaded_audio():
    audio_file = request.files.get("audio_file")
    selected_call_type = str(request.form.get("call_type", "default")).strip().lower()

    if audio_file is None or not audio_file.filename:
        return jsonify({"success": False, "error": "Please choose an audio file to upload."}), 400

    file_extension = Path(audio_file.filename).suffix.lower()
    if file_extension not in ALLOWED_UPLOAD_EXTENSIONS:
        return jsonify({"success": False, "error": "Unsupported file type. Upload wav, mp3, flac, ogg, or m4a."}), 400

    normalized_call_type, _ = get_call_config(selected_call_type)
    upload_name = f"{Path(audio_file.filename).stem}_{uuid4().hex[:8]}{file_extension}"
    upload_path = UPLOAD_FOLDER / upload_name
    audio_file.save(upload_path)

    try:
        cleaned_name, metrics, selected_preset = clean_uploaded_clip(upload_path, normalized_call_type)
        cleaned_path = UPLOAD_FOLDER / cleaned_name

        original_spectrogram_url = save_spectrogram(upload_path, f"{upload_path.stem}_original")
        cleaned_spectrogram_url = save_spectrogram(cleaned_path, f"{cleaned_path.stem}_cleaned")

        return jsonify(
            {
                "success": True,
                "message": "Upload processed successfully.",
                "call_type": normalized_call_type,
                "selected_preset": selected_preset,
                "original_audio_url": url_for("static", filename=f"uploads/{upload_name}"),
                "cleaned_audio_url": url_for("static", filename=f"uploads/{cleaned_name}"),
                "original_spectrogram_url": original_spectrogram_url,
                "cleaned_spectrogram_url": cleaned_spectrogram_url,
                "metrics": {
                    "Target Retention %": round(float(metrics["target_band_retention_pct"]), 2),
                    "Machine Suppression %": round(float(metrics["machine_band_suppression_pct"]), 2),
                    "Target/Machine Gain dB": round(float(metrics["target_machine_ratio_gain_db"]), 2),
                    "Target Similarity": round(float(metrics["target_band_cosine_similarity"]), 3),
                },
            }
        )
    except Exception as exc:  # pragma: no cover
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
