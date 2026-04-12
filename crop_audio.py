from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from pydub import AudioSegment


CSV_FILE = "Audio_Files_Master.csv"
FOLDER_IN = "Elephant_Raw_Audio"
CALL_ONLY_OUT = "Elephant_Training_Snippets"
CALL_CONTEXT_OUT = "Elephant_Call_Context"
SAFE_NOISE_OUT = "Elephant_Safe_Noise"
CONTEXT_BUFFER_MS = 2000
MIN_NOISE_MS = 1500


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / FOLDER_IN
CALL_ONLY_DIR = BASE_DIR / CALL_ONLY_OUT
CALL_CONTEXT_DIR = BASE_DIR / CALL_CONTEXT_OUT
SAFE_NOISE_DIR = BASE_DIR / SAFE_NOISE_OUT

for folder in (CALL_ONLY_DIR, CALL_CONTEXT_DIR, SAFE_NOISE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def to_ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000))


def export_clip(audio: AudioSegment, start_ms: int, end_ms: int, output_path: Path) -> None:
    if end_ms <= start_ms:
        return
    clip = audio[start_ms:end_ms]
    clip.export(output_path, format="wav")


def build_safe_noise_window(
    row: pd.Series,
    prev_end_ms: int,
    next_start_ms: int,
    audio_length_ms: int,
) -> tuple[int, int] | None:
    start_ms = to_ms(row["Start_time"])
    end_ms = to_ms(row["End_time"])

    left_gap = start_ms - prev_end_ms
    right_gap = next_start_ms - end_ms

    if left_gap >= MIN_NOISE_MS and left_gap >= right_gap:
        return prev_end_ms, start_ms
    if right_gap >= MIN_NOISE_MS:
        return end_ms, next_start_ms

    return None


def main() -> None:
    print(f"Working Directory: {BASE_DIR}")
    print(f"Looking for audio in: {INPUT_DIR}")

    data = pd.read_csv(BASE_DIR / CSV_FILE)
    data["Sound_file"] = data["Sound_file"].astype(str).str.strip()
    data = data.sort_values(["Sound_file", "Start_time", "End_time"]).reset_index(drop=True)

    for sound_file, group in data.groupby("Sound_file", sort=False):
        source_path = INPUT_DIR / sound_file
        if not source_path.exists():
            print(f"File not found: {source_path}")
            continue

        try:
            audio = AudioSegment.from_file(source_path)
        except Exception as exc:
            print(f"Error loading {sound_file}: {exc}")
            continue

        audio_length_ms = len(audio)
        group = group.reset_index(drop=True)

        for idx, row in group.iterrows():
            selection = int(row["Selection"])
            start_ms = max(0, to_ms(row["Start_time"]))
            end_ms = min(audio_length_ms, to_ms(row["End_time"]))

            if end_ms <= start_ms:
                print(f"Skipping call_{selection}: invalid time window")
                continue

            call_only_path = CALL_ONLY_DIR / f"call_{selection}.wav"
            export_clip(audio, start_ms, end_ms, call_only_path)

            context_start_ms = max(0, start_ms - CONTEXT_BUFFER_MS)
            context_end_ms = min(audio_length_ms, end_ms + CONTEXT_BUFFER_MS)
            context_path = CALL_CONTEXT_DIR / f"call_{selection}_context.wav"
            export_clip(audio, context_start_ms, context_end_ms, context_path)

            prev_end_ms = 0 if idx == 0 else to_ms(group.iloc[idx - 1]["End_time"])
            next_start_ms = audio_length_ms if idx == len(group) - 1 else to_ms(group.iloc[idx + 1]["Start_time"])
            noise_window = build_safe_noise_window(row, prev_end_ms, next_start_ms, audio_length_ms)

            if noise_window is not None:
                noise_start_ms, noise_end_ms = noise_window
                noise_path = SAFE_NOISE_DIR / f"call_{selection}_noise.wav"
                export_clip(audio, noise_start_ms, noise_end_ms, noise_path)
                noise_status = "safe noise saved"
            else:
                noise_status = "no safe noise gap"

            print(
                f"Created call_{selection}.wav | context: {context_path.name} | {noise_status}"
            )

    print("\n--- Process Finished ---")


if __name__ == "__main__":
    main()
