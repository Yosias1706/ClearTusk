# ClearTusk

**Recover elephant vocalisations buried under aircraft, vehicle, and generator noise — then prove how well it worked.**

Elephants talk mostly below 150 Hz. So do aeroplanes, Land Cruisers, and camp
generators. ClearTusk is a full bioacoustic toolkit around that collision: it
**detects** where calls occur in a recording, **isolates** them from machine
noise, and **measures** the result against reference-free quality metrics that
are stored for every run, so a change to the DSP can be shown to help or hurt.

<p align="center">
  <img src="docs/screenshots/overview.png" alt="ClearTusk overview page" width="900">
</p>

---

## Results

Measured on the bundled corpus of **212 annotated elephant calls** cut from
**38 field recordings** (1.8 hours of raw audio), re-run end to end by
`cleartusk benchmark`:

| Metric | Result | What it means |
|---|---|---|
| Call retained | **76.3 %** | Mean protected-band energy surviving the clean |
| Machine noise removed | **91.3 %** | Mean energy cut above the protected band |
| Call-to-noise gain | **+12.6 dB** | How much further the call stands out afterwards |
| Spectral fidelity | **0.964** | Cosine similarity of the in-band spectrum (1.0 = untouched) |
| Peak still in band | **85.8 %** | Clips whose dominant frequency stays inside the call band |
| Batch throughput | **125× real time** | 13.6 min of audio in 6.5 s across 8 worker processes |
| Interactive latency | **~1.2 s p50** | Detect → clean → render → persist, for a typical upload |

**Call-activity detector** — a gradient-boosted classifier over 56 spectral,
temporal, and cepstral features, trained on 1,731 windows (729 call / 1,002
noise) and validated with **5-fold `StratifiedGroupKFold` grouped by source
recording**, so no field site appears in both halves of a fold:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 81.7 % | 81.0 % | 76.0 % | 0.765 | **0.903** |

Every one of these numbers is computed from the database at request time and
rendered on `/analytics` — nothing in the UI is hard-coded.

<p align="center">
  <img src="docs/screenshots/studio.png" alt="Processing studio with detection timeline and spectrograms" width="900">
</p>

---

## What it does

1. **Detect.** A sliding 1.5 s window sweeps the recording; the classifier
   scores each window, and above-threshold windows merge into call events with
   a confidence, a peak frequency, and a predicted call type.
2. **Profile.** The dominant event chooses the call profile — rumble
   (10–150 Hz), roar (40–300 Hz), trumpet (80–450 Hz), or a wide default —
   which fixes the band the cleaner must protect. A user can override it.
3. **Isolate.** For each preset in that profile: harmonic/percussive
   separation, a harmonic-ridge mask, an optional noise gate built from a
   call-free window of the *same* recording, a frequency weight that boosts the
   protected band and tapers (never hard-cuts) above it, then a controlled
   re-injection of the original in-band call. In-band energy is capped at the
   original so retention can never be inflated.
4. **Score and store.** Every candidate is scored; the winner is written to
   PostgreSQL with its metrics, detections, artifacts, and timings.

### The metrics, precisely

There is no clean ground truth for a field recording, so quality is expressed
as *reference-free proxies* — each one defined in `cleartusk/audio/metrics.py`:

- **Target retention %** — in-band energy of the cleaned clip over in-band
  energy of the original, capped at 100 %.
- **Machine suppression %** — the same ratio in the noise band, which always
  starts at least 25 Hz above the protected band so preserved call energy can
  never be counted as removed noise.
- **Call-to-noise gain (dB)** — change in the band-energy ratio, the single
  number that says "does the call stand out more now?".
- **Spectral fidelity** — cosine similarity between the time-averaged in-band
  spectra, which catches a denoiser that keeps the energy but wrecks the shape.
- **Candidate score** — the weighted combination used to pick a preset. It
  penalises retention below a per-profile floor, so a preset cannot win by
  deleting the call.

---

## Quickstart

```bash
git clone <this repo> && cd ClearTusk
make install        # venv + pip install -e ".[postgres,server,dev]"
make bootstrap      # schema, corpus ingest, detector training, benchmark sweep
make serve          # http://127.0.0.1:5000
```

That works with **no configuration at all**: without a `.env` file ClearTusk
creates a SQLite database under `data/runtime/` and every feature — detection,
cleaning, analytics, the API — behaves identically. `make bootstrap` is
idempotent and prints the scorecard it produces.

### Choosing a database

There is one database layer — a SQLAlchemy URL in `CLEARTUSK_DATABASE_URL` —
with two supported backends:

| Backend | URL | Used for |
|---|---|---|
| **SQLite** (default, no setup) | unset → a file in `data/runtime/` | Tests, a quick demo, a fresh clone |
| **PostgreSQL** | `postgresql+psycopg://user:pw@host:5432/cleartusk`, or `postgresql+psycopg:///cleartusk?host=/tmp` for a local unix socket | Everything else |

Docker Compose is not a third option; it is simply one way to *run* the
PostgreSQL in row two, and it sets the URL for you.

The fallback is not a second code path that might rot: the **entire test suite
runs against both backends in CI**, driven by `CLEARTUSK_TEST_DATABASE_URL`,
and Alembic migrations are applied, rolled back, and re-applied on PostgreSQL
on every push. Run the suite against your own server the same way:

```bash
createdb cleartusk_test
CLEARTUSK_TEST_DATABASE_URL="postgresql+psycopg:///cleartusk_test?host=/tmp" pytest
```

A local PostgreSQL installed via Homebrew usually authenticates over a unix
socket as your own user, so the socket URL above is the one that works; the
`user:password@localhost` form in `.env.example` is the Docker Compose service.
`.env.example` documents every setting — copy it only if you want to change one.

Before anything public, apply migrations and set a real secret key:

```bash
make migrate                            # alembic upgrade head
export CLEARTUSK_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

### Docker

```bash
docker compose up --build    # PostgreSQL 16 + gunicorn on :8000, migrations applied
```

This brings up the whole stack — database, migrations, and the app behind
gunicorn — with no Python toolchain on the host. Use it to hand the project to
someone else, to reproduce the deployment locally, or on a machine with no
PostgreSQL installed. If you already run PostgreSQL and develop with
`make serve`, you never need it.

---

## Command line

| Command | Purpose |
|---|---|
| `cleartusk config` | Print every resolved path and the database backend |
| `cleartusk init-db` / `alembic upgrade head` | Create or migrate the schema |
| `cleartusk prepare-corpus` | Cut call / context / safe-noise clips from the raw recordings |
| `cleartusk ingest-corpus` | Load the annotation table into the database |
| `cleartusk train-detector` | Train and register the call detector with a CV scorecard |
| `cleartusk benchmark --csv` | Re-clean the whole corpus in parallel, store the aggregates |
| `cleartusk clean FILE --call-type auto` | Run one file through the pipeline |
| `cleartusk stats` | Print the analytics dashboard in the terminal |
| `cleartusk bootstrap` | All of the above, in order |
| `cleartusk serve` | Development web server |

## HTTP API

| Endpoint | Description |
|---|---|
| `POST /api/v1/process` | Upload audio (`audio_file`, optional `call_type`) → metrics, detections, asset URLs |
| `GET /api/v1/runs` | Paginated run history (`page`, `page_size`, `call_type`) |
| `GET /api/v1/runs/<id>` | One run with metrics, detected events, and artifacts |
| `GET /api/v1/analytics` | The exact payload behind the dashboard |
| `GET /api/v1/benchmarks` | Benchmark history and the latest aggregates |
| `GET /api/v1/detector` | Active model version and its cross-validated scorecard |
| `GET /api/v1/call-types` | Call profiles, protected bands, and presets |
| `GET /healthz` | Liveness, database backend, detector availability |

```bash
curl -F "audio_file=@recording.wav" -F "call_type=auto" \
     http://127.0.0.1:5000/api/v1/process | jq '.metrics, .detection.events'
```

---

## Architecture

```text
cleartusk/
  config.py             Env-driven settings; every path derives from here
  logging_config.py     Shared logging setup
  cli.py                Click CLI (the entry point `cleartusk`)
  audio/
    presets.py          Call profiles and denoiser presets (frozen dataclasses)
    io.py               Loading, writing, resampling, probing, hashing
    denoise.py          The harmonic isolation pipeline
    metrics.py          Reference-free quality metrics and candidate scoring
    features.py         56-feature descriptor + spectral call-type heuristic
    detection.py        Detector training, checkpointing, sliding-window sweep
    spectrogram.py      Shared-scale before/after rendering
  db/
    __init__.py         Engine, session scope, schema helpers
    models.py           SQLAlchemy 2.0 ORM (9 tables)
  services/
    corpus.py           Clip preparation and annotation ingest
    training.py         Feature matrix assembly, CV, model registration
    pipeline.py         Upload → detect → clean → render → persist
    benchmark.py        Parallel corpus sweep and aggregation
    analytics.py        Every dashboard statistic, computed in SQL
    storage.py          Keyed media storage with traversal guards
    demo.py             Cached landing-page showcase
  web/
    __init__.py         App factory, per-request session, error handling
    blueprints/         site.py (HTML) and api.py (JSON)
    templates/  static/ Jinja templates, design-system CSS, Chart.js views
alembic/                Migrations (verified against PostgreSQL in CI)
tests/                  65 tests, 85 % statement coverage
data/
  corpus/               raw/ calls/ noise/ context/ annotations.csv
  processed/cleaned/    Batch-cleaned outputs
  runtime/              Uploads, spectrograms, models, reports (gitignored)
```

**Design decisions worth calling out**

- **No hard-coded paths.** Everything resolves from `Settings`, which finds the
  project root by marker file and lets `CLEARTUSK_DATA_DIR`,
  `CLEARTUSK_CORPUS_DIR`, and `CLEARTUSK_RUNTIME_DIR` be moved independently.
  A test asserts that no module contains an absolute user path.
- **Generated artifacts never touch the source tree.** Uploads, spectrograms,
  models, and reports live under the runtime directory and are served through
  one guarded route that refuses keys escaping the storage root.
- **One source of truth for statistics.** The HTML dashboard, the JSON API, and
  `cleartusk stats` all call the same `AnalyticsService`, so they cannot drift.
- **Portable schema.** Backend-neutral column types mean the same models run on
  PostgreSQL and SQLite; CI exercises the migrations on PostgreSQL both ways.
- **Honest evaluation.** Detector folds are grouped by source recording, and
  training windows are the same length as inference windows so clip duration
  cannot leak into the model.
- **Charts built to a spec.** The categorical palette is validated for
  colour-vision deficiency separation and contrast; every chart carries a
  legend or a data table, and no chart uses two y-axes.

### Data model

| Table | Holds |
|---|---|
| `recordings` | Uploaded or corpus audio: hash, duration, sample rate, storage key |
| `processing_runs` | One pipeline execution: profile, preset, timings, artifacts |
| `run_metrics` | The quality vector for a run (1:1) |
| `call_detections` | Detected events with confidence and predicted type |
| `corpus_clips` | The annotation table, plus derived noise source and coverage |
| `benchmark_runs` / `benchmark_results` | Sweep aggregates and per-clip outcomes |
| `model_versions` | Detector checkpoints with their cross-validated scorecards |
| `contact_messages` | Inbound messages from the contact form |

<p align="center">
  <img src="docs/screenshots/analytics.png" alt="Analytics dashboard" width="900">
</p>

---

## Development

```bash
make test        # pytest
make coverage    # pytest with a coverage report
make lint        # ruff
make revision m="add table"   # autogenerate a migration
```

The suite builds a synthetic corpus in a temporary directory, so it never
touches the real dataset and needs no database server. CI additionally runs the
migrations against PostgreSQL 16, upgrade → downgrade → upgrade.

## Configuration

Every setting is an environment variable prefixed `CLEARTUSK_` (see
`.env.example`). The ones worth knowing:

| Variable | Default | Purpose |
|---|---|---|
| `CLEARTUSK_DATABASE_URL` | SQLite under `data/runtime/` | Any SQLAlchemy URL |
| `CLEARTUSK_DATA_DIR` / `CLEARTUSK_CORPUS_DIR` / `CLEARTUSK_RUNTIME_DIR` | derived from the project root | Relocate datasets and artifacts |
| `CLEARTUSK_MAX_UPLOAD_MB` | `64` | Upload size limit |
| `CLEARTUSK_SAMPLE_RATE` | `2000` | Analysis rate — the whole repertoire is below 1 kHz |
| `CLEARTUSK_DETECTOR_THRESHOLD` | `0.6` | Detection sensitivity |
| `CLEARTUSK_RETAIN_RUN_ARTIFACTS` | `200` | How many upload/cleaned pairs to keep on disk |

---

## Limitations

- The cleaner is a **signal-processing baseline**, not a learned separator. It
  reduces machine noise inside a protected band; it does not reconstruct call
  energy that the noise destroyed.
- The corpus is **94 % rumble** (200 of 212 calls). Rumble results are the
  trustworthy ones; the trumpet and roar profiles are tuned but thinly
  evidenced, and the call-*type* step is therefore a documented spectral
  heuristic rather than a classifier trained on those labels.
- Quality metrics are proxies. A perfect score means in-band energy survived
  and out-of-band energy did not; it is not a substitute for a biologist
  listening to the clip.
- Detector recall (76 %) trails precision, so quiet or partially masked calls
  are still missed; the sweep is intentionally tuned against false positives.

## Roadmap

- Mask-based learned separation trained against the harmonic baseline
- More trumpet/roar annotations, then a genuine call-type classifier
- Object storage backend behind the existing `MediaStorage` interface
- Streaming/long-recording mode with chunked detection and partial results

## Credits

Recordings and annotations come from long-term elephant field research; the raw
audio is redistributed here for research and demonstration purposes. Code is
MIT licensed.
