import csv
from pathlib import Path
from uuid import uuid4

import librosa  # noqa: F401
import matplotlib
import numpy as np  # noqa: F401
import soundfile as sf
from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for

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
SPECTROGRAM_FOLDER = STATIC_DIR / "spectrograms"
ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SPECTROGRAM_MAX_HZ = 700
SPECTROGRAM_N_FFT = 8192
CONTACT_MESSAGES_FILE = BASE_DIR / "contact_messages.csv"
HOME_PAGE_IMAGES = {
    "plane.png": "No airplane noise",
    "car.png": "No vehicle noise",
    "generator.png": "No generator noise",
}
SITE_ASSET_FILES = {"elephant.png", *HOME_PAGE_IMAGES.keys()}
HOME_PAGE_SPECTROGRAM_EXAMPLES = [
    {
        "title": "Original Spectrogram 1",
        "source": BASE_DIR / "Elephant_Training_Snippets" / "call_4.wav",
        "output": "home_call_4_original.png",
    },
    {
        "title": "Original Spectrogram 2",
        "source": BASE_DIR / "Elephant_Training_Snippets" / "call_3.wav",
        "output": "home_call_3_original.png",
    },
    {
        "title": "Cleaned Spectrogram 1",
        "source": BASE_DIR / "Harmonic_Isolated_Clips_1_1000Hz" / "call_4.wav",
        "output": "home_call_4_cleaned.png",
    },
    {
        "title": "Cleaned Spectrogram 2",
        "source": BASE_DIR / "Harmonic_Isolated_Clips_1_1000Hz" / "call_3.wav",
        "output": "home_call_3_cleaned.png",
    },
]

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
SPECTROGRAM_FOLDER.mkdir(parents=True, exist_ok=True)


def get_call_config(call_type: str) -> tuple[str, dict[str, object]]:
    normalized = str(call_type).strip().lower() or "default"
    if normalized not in CALL_TYPE_CONFIGS:
        normalized = "default"
    return normalized, CALL_TYPE_CONFIGS.get(normalized, DEFAULT_CALL_CONFIG)


def render_spectrogram_image(audio_path: Path, output_path: Path) -> None:
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


def save_spectrogram(audio_path: Path, prefix: str) -> str:
    output_name = f"{prefix}_{uuid4().hex[:8]}.png"
    output_path = SPECTROGRAM_FOLDER / output_name
    render_spectrogram_image(audio_path, output_path)

    return url_for("static", filename=f"spectrograms/{output_name}")


def ensure_home_spectrogram_examples() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for example in HOME_PAGE_SPECTROGRAM_EXAMPLES:
        output_path = SPECTROGRAM_FOLDER / str(example["output"])
        source_path = Path(example["source"])
        if source_path.exists():
            render_spectrogram_image(source_path, output_path)
            examples.append(
                {
                    "title": str(example["title"]),
                    "image_url": url_for("static", filename=f"spectrograms/{output_path.name}"),
                }
            )
    return examples


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
    home_examples = [
        {
            "title": title,
            "image_url": url_for("home_asset", filename=filename),
        }
        for filename, title in HOME_PAGE_IMAGES.items()
    ]
    spectrogram_examples = ensure_home_spectrogram_examples()
    return render_template(
        "home.html",
        home_examples=home_examples,
        spectrogram_examples=spectrogram_examples,
    )


@app.route("/home-assets/<path:filename>")
def home_asset(filename: str):
    if filename not in SITE_ASSET_FILES:
        abort(404)
    return send_from_directory(BASE_DIR, filename)


@app.route("/dashboard")
def index():
    return render_template("index.html", call_types=["rumble", "trumpet", "roar", "default"])


@app.route("/contact", methods=["GET", "POST"])
def contact():
    status = None
    form_data = {
        "name": "",
        "email": "",
        "subject": "",
        "message": "",
    }

    if request.method == "POST":
        form_data = {
            "name": str(request.form.get("name", "")).strip(),
            "email": str(request.form.get("email", "")).strip(),
            "subject": str(request.form.get("subject", "")).strip(),
            "message": str(request.form.get("message", "")).strip(),
        }

        if not all(form_data.values()):
            status = {
                "kind": "error",
                "message": "Please fill out all four fields before sending your message.",
            }
        else:
            is_new_file = not CONTACT_MESSAGES_FILE.exists()
            with CONTACT_MESSAGES_FILE.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["name", "email", "subject", "message"])
                if is_new_file:
                    writer.writeheader()
                writer.writerow(form_data)

            status = {
                "kind": "success",
                "message": "Your message has been saved. We’ll be able to review it from this project workspace.",
            }
            form_data = {
                "name": "",
                "email": "",
                "subject": "",
                "message": "",
            }

    return render_template("contact.html", status=status, form_data=form_data)


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
                    "Target Retention % (how much of the elephant sound was preserved)": round(
                        float(metrics["target_band_retention_pct"]), 2
                    ),
                    "Machine Suppression % (how much machine noise was removed)": round(
                        float(metrics["machine_band_suppression_pct"]), 2
                    ),
                    "Target/Machine Gain dB (how much more the elephant stands out)": round(
                        float(metrics["target_machine_ratio_gain_db"]), 2
                    ),
                    "Target Spectral Similarity (how closely the cleaned call matches the original sound profile)": round(
                        float(metrics["target_band_cosine_similarity"]), 3
                    ),
                },
            }
        )
    except Exception as exc:  # pragma: no cover
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
