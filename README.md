# ClearTusk

ClearTusk is a Flask-based prototype for cleaning elephant recordings that contain overlapping machine noise such as airplanes, vehicles, and generators.

The app lets a user:
- upload an elephant recording
- choose the call type
- generate a cleaned version of the audio
- compare original and cleaned spectrograms
- review simple preservation and noise-reduction metrics

## What This Project Does

This repository combines:
- a web app for uploading and reviewing recordings
- a custom Python audio-cleaning pipeline in `harmonic_cleaner.py`
- a small research workflow for preparing training clips and measuring results

The current cleaner is a signal-processing baseline, not a trained machine learning model. It uses harmonic separation, spectrogram masking, and protected call-band preservation to reduce machine noise while trying to keep the elephant call intact.

## Tech Stack

- Python
- Flask
- HTML, CSS, JavaScript
- Bootstrap 5
- librosa
- NumPy
- SciPy
- pandas
- soundfile
- matplotlib

## Project Structure

```text
app.py                         Flask app and upload pipeline
harmonic_cleaner.py            Main denoising and evaluation pipeline
crop_audio.py                  Script for creating cropped call/noise/context clips
Audio_Files_Master.csv         Master annotation file
metrics_report.csv             Latest aggregate and per-clip metrics report

templates/
  layout.html                  Shared page layout and navbar
  home.html                    Landing page
  index.html                   Main audio intake dashboard
  contact.html                 Contact form page

static/
  css/styles.css               Site styling
  js/main.js                   Frontend upload/results behavior
  uploads/                     Uploaded and cleaned audio from the web app
  spectrograms/                Generated spectrogram images

Elephant_Training_Snippets/    Cropped elephant call clips
Elephant_Safe_Noise/           Nearby non-call noise clips
Elephant_Call_Context/         Call clips with local context
Elephant_Raw_Audio/            Source recordings
Harmonic_Isolated_Clips_1_1000Hz/
                               Batch-cleaned training outputs
```

## How To Run The Web App

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## How To Use The App

1. Open the home page.
2. Go to `Audio Intake`.
3. Upload an audio file.
4. Select the call type:
   - `rumble`
   - `trumpet`
   - `roar`
   - `default`
5. Submit the recording.
6. Review:
   - original audio
   - cleaned audio
   - original spectrogram
   - cleaned spectrogram
   - output metrics

## Running The Offline Cleaner

To run the batch cleaner over the prepared training clips:

```bash
source .venv/bin/activate
python3 harmonic_cleaner.py
```

This will:
- process the clips in `Elephant_Training_Snippets/`
- use `Elephant_Safe_Noise/` and `Elephant_Call_Context/` when available
- save cleaned outputs into `Harmonic_Isolated_Clips_1_1000Hz/`
- update `metrics_report.csv`

## Main Metrics

The app and report currently display:

- `Target Retention %`
  How much of the elephant sound was preserved in the protected call band.

- `Machine Suppression %`
  How much sound was reduced in the higher-frequency machine-noise band.

- `Target/Machine Gain dB`
  How much more the elephant call stands out relative to machine noise after cleaning.

- `Target Spectral Similarity`
  How closely the cleaned call matches the original target-band sound profile.

## Notes

- This project currently stores uploads, spectrograms, and contact messages locally on disk.
- There is no database or cloud storage in the current prototype.
- The current live app still asks the user to choose the call type manually.
- The cleaner is strongest on `rumble` clips because the dataset is heavier in rumble examples than in `roar` or `trumpet`.

## Future Improvements

- Move file storage to cloud storage
- Add automatic call-type detection
- Save processing history and past runs
- Expand the dataset, especially for roar and trumpet calls
- Explore machine learning approaches beyond the current signal-processing baseline
