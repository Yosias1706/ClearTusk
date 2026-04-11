from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import librosa  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd
import soundfile as sf
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, url_for
from scipy import signal  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
AUDIO_FOLDER = STATIC_DIR / "audio"
METADATA_FILE = BASE_DIR / "metadata.csv"

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
app.config["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)


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


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def index():
    metadata = load_metadata()
    return render_template("index.html", metadata=metadata)


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


if __name__ == "__main__":
    app.run(debug=True)
