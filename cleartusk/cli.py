"""``cleartusk`` command line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from cleartusk import __version__
from cleartusk.config import get_settings
from cleartusk.db import database_is_reachable, init_database, session_scope
from cleartusk.logging_config import configure_logging

logger = logging.getLogger("cleartusk.cli")


def _echo_kv(title: str, rows: dict[str, object]) -> None:
    click.secho(title, bold=True)
    width = max((len(str(key)) for key in rows), default=0)
    for key, value in rows.items():
        click.echo(f"  {str(key).ljust(width)}  {value}")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="cleartusk")
@click.option("--log-level", default=None, help="Override the configured log level.")
def cli(log_level: str | None) -> None:
    """Elephant call isolation, detection, and analytics."""
    settings = get_settings()
    if log_level:
        settings.log_level = log_level
    configure_logging(settings, force=True)


@cli.command()
def config() -> None:
    """Show the resolved configuration and paths."""
    settings = get_settings()
    _echo_kv(
        "ClearTusk configuration",
        {
            "environment": settings.environment,
            "project root": settings.project_root,
            "data dir": settings.resolved_data_dir,
            "corpus dir": settings.resolved_corpus_dir,
            "runtime dir": settings.resolved_runtime_dir,
            "database": settings.resolved_database_url.split("@")[-1],
            "backend": settings.database_backend,
            "detector model": settings.detector_model_path,
            "sample rate": f"{settings.sample_rate} Hz",
            "upload limit": f"{settings.max_upload_mb} MB",
        },
    )
    click.echo()
    click.echo("database reachable: " + ("yes" if database_is_reachable() else "no"))


@cli.command("init-db")
@click.option("--drop", is_flag=True, help="Drop existing tables first (destructive).")
def init_db(drop: bool) -> None:
    """Create the database schema."""
    if drop and not click.confirm("This deletes every stored run. Continue?"):
        raise click.Abort()
    init_database(drop=drop)
    click.secho(f"schema ready on {get_settings().database_backend}", fg="green")


@cli.command("prepare-corpus")
@click.option("--overwrite", is_flag=True, help="Re-cut clips that already exist.")
def prepare_corpus(overwrite: bool) -> None:
    """Cut call, context, and safe-noise clips from the raw recordings."""
    from cleartusk.services.corpus import prepare_clips

    counts = prepare_clips(overwrite=overwrite)
    _echo_kv("Corpus prepared", counts)


@cli.command("ingest-corpus")
def ingest_corpus_command() -> None:
    """Load the annotation table into the database."""
    from cleartusk.services.corpus import ingest_corpus

    init_database()
    with session_scope() as session:
        count = ingest_corpus(session)
    click.secho(f"ingested {count} annotated clips", fg="green")


@cli.command("train-detector")
@click.option("--folds", default=5, show_default=True, help="Cross-validation folds.")
def train_detector_command(folds: int) -> None:
    """Train the call-activity detector and register its scorecard."""
    from cleartusk.services.training import train_and_register

    init_database()
    with session_scope() as session:
        report = train_and_register(session, folds=folds)

    _echo_kv(
        "Detector trained",
        {
            "algorithm": report.algorithm,
            "windows": f"{report.sample_count} ({report.positive_count} call / {report.negative_count} noise)",
            "features": report.feature_count,
            "validation": report.cv_strategy,
            "accuracy": f"{report.accuracy:.3f}",
            "precision": f"{report.precision:.3f}",
            "recall": f"{report.recall:.3f}",
            "f1": f"{report.f1:.3f}",
            "roc auc": f"{report.roc_auc:.3f}",
            "model": get_settings().detector_model_path,
        },
    )


@cli.command()
@click.option("--label", default="corpus sweep", show_default=True)
@click.option("--limit", type=int, default=None, help="Only process the first N clips.")
@click.option("--workers", type=int, default=None, help="Parallel worker processes.")
@click.option(
    "--export-audio/--no-export-audio", default=False, help="Write cleaned clips to data/processed."
)
@click.option(
    "--noise-reference/--no-noise-reference", default=True, help="Use per-recording noise references."
)
@click.option("--csv", "write_csv", is_flag=True, help="Also write a per-clip CSV report.")
def benchmark(
    label: str,
    limit: int | None,
    workers: int | None,
    export_audio: bool,
    noise_reference: bool,
    write_csv: bool,
) -> None:
    """Re-clean the whole corpus and record the aggregate scorecard."""
    from cleartusk.services.benchmark import export_report, run_benchmark

    init_database()
    with session_scope() as session:
        result = run_benchmark(
            session,
            label=label,
            limit=limit,
            workers=workers,
            use_noise_reference=noise_reference,
            export_audio=export_audio,
            progress=True,
        )
        aggregates = result.aggregates
        metrics = aggregates["metrics"]
        _echo_kv(
            f"Benchmark '{label}'",
            {
                "clips": result.clip_count,
                "audio": f"{result.total_audio_seconds / 60:.1f} min",
                "wall time": f"{result.wall_seconds:.1f} s",
                "throughput": f"{result.realtime_factor:.1f}× real time on {aggregates['workers']} workers",
                "retention": f"{metrics['target_retention_pct']['mean']:.1f}% (p10 {metrics['target_retention_pct']['p10']:.1f}%)",
                "suppression": f"{metrics['machine_suppression_pct']['mean']:.1f}%",
                "gain": f"+{metrics['target_machine_gain_db']['mean']:.2f} dB",
                "fidelity": f"{metrics['spectral_fidelity']['mean']:.3f}",
                "peak in band": f"{aggregates['peak_in_target_band_pct']:.1f}%",
            },
        )
        if write_csv:
            click.echo(f"report: {export_report(result)}")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--call-type", default="auto", show_default=True, help="auto, rumble, trumpet, roar, default.")
@click.option("--json", "as_json", is_flag=True, help="Print the full result as JSON.")
def clean(audio_path: Path, call_type: str, as_json: bool) -> None:
    """Clean one recording through the full pipeline and store the run."""
    from cleartusk.services.pipeline import CleaningPipeline

    init_database()
    with session_scope() as session:
        pipeline = CleaningPipeline(session)
        outcome = pipeline.process_file(audio_path, call_type)
        payload = outcome.as_dict()
        cleaned_path = pipeline.storage.path_for(outcome.cleaned_key)

    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return

    metrics = outcome.metrics
    _echo_kv(
        f"Cleaned {audio_path.name}",
        {
            "profile": f"{outcome.call_type} ({outcome.call_type_source})",
            "preset": f"{outcome.preset} (best of {outcome.candidates_evaluated})",
            "events": f"{len(outcome.detection.events)} detected, peak {outcome.detection.peak_confidence:.2f}",
            "retention": f"{metrics.target_retention_pct:.1f}%",
            "suppression": f"{metrics.machine_suppression_pct:.1f}%",
            "gain": f"+{metrics.target_machine_gain_db:.2f} dB",
            "fidelity": f"{metrics.spectral_fidelity:.3f}",
            "speed": f"{outcome.duration_ms:.0f} ms ({outcome.realtime_factor:.1f}× real time)",
            "cleaned file": cleaned_path,
            "run id": outcome.run_public_id,
        },
    )


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Print the full analytics payload as JSON.")
def stats(as_json: bool) -> None:
    """Print the analytics summary shown on the dashboard."""
    from cleartusk.services.analytics import AnalyticsService

    init_database()
    with session_scope() as session:
        data = AnalyticsService(session).dashboard()

    if as_json:
        click.echo(json.dumps(data, indent=2, default=str))
        return

    overview, performance = data["overview"], data["performance"]
    _echo_kv(
        "Corpus",
        {
            "annotated calls": overview["corpus_clips"],
            "source recordings": overview["corpus_recordings"],
            "corpus duration": f"{overview['corpus_minutes']} min",
        },
    )
    click.echo()
    _echo_kv(
        "Quality (latest benchmark)",
        {
            "clips": overview["benchmark_clip_count"],
            "call retained": f"{overview['avg_target_retention_pct']}%",
            "machine removed": f"{overview['avg_machine_suppression_pct']}%",
            "call-to-noise gain": f"+{overview['avg_gain_db']} dB",
            "spectral fidelity": overview["avg_spectral_fidelity"],
        },
    )
    click.echo()
    _echo_kv(
        "Throughput",
        {
            "batch": f"{performance['batch_realtime_factor']}× real time on {performance['batch_workers']} workers",
            "interactive p50": f"{performance['latency_p50_ms']:.0f} ms",
            "interactive p95": f"{performance['latency_p95_ms']:.0f} ms",
            "runs": f"{overview['runs_total']} ({overview['success_rate_pct']}% succeeded)",
            "audio processed": f"{overview['audio_processed_minutes']} min",
        },
    )
    detector = data.get("detector")
    if detector:
        click.echo()
        _echo_kv(
            "Call detector",
            {
                "algorithm": detector["algorithm"],
                "validation": detector["cv_strategy"],
                "accuracy": detector["accuracy"],
                "f1": detector["f1"],
                "roc auc": detector["roc_auc"],
            },
        )


@cli.command()
@click.option("--skip-benchmark", is_flag=True, help="Skip the corpus sweep.")
@click.option("--benchmark-limit", type=int, default=None, help="Benchmark only the first N clips.")
def bootstrap(skip_benchmark: bool, benchmark_limit: int | None) -> None:
    """One-shot setup: schema, corpus ingest, detector training, benchmark sweep."""
    from cleartusk.services.benchmark import run_benchmark
    from cleartusk.services.corpus import corpus_is_present, ingest_corpus, prepare_clips
    from cleartusk.services.training import train_and_register

    settings = get_settings()
    settings.ensure_runtime_dirs()
    init_database()
    click.secho(f"✓ schema ready ({settings.database_backend})", fg="green")

    if not corpus_is_present():
        click.echo("cutting corpus clips from the raw recordings…")
        prepare_clips()
    with session_scope() as session:
        count = ingest_corpus(session)
    click.secho(f"✓ {count} annotated clips ingested", fg="green")

    with session_scope() as session:
        report = train_and_register(session)
    click.secho(
        f"✓ detector trained — ROC-AUC {report.roc_auc:.3f}, F1 {report.f1:.3f} "
        f"({report.cv_folds}-fold, grouped by recording)",
        fg="green",
    )

    if not skip_benchmark:
        with session_scope() as session:
            result = run_benchmark(session, label="bootstrap sweep", limit=benchmark_limit, progress=True)
        metrics = result.aggregates["metrics"]
        click.secho(
            f"✓ benchmark over {result.clip_count} clips — "
            f"{metrics['target_retention_pct']['mean']:.1f}% retained, "
            f"{metrics['machine_suppression_pct']['mean']:.1f}% suppressed, "
            f"+{metrics['target_machine_gain_db']['mean']:.2f} dB",
            fg="green",
        )

    click.echo()
    click.secho("Ready. Start the app with `cleartusk serve`.", bold=True)


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=5000, show_default=True, type=int)
@click.option("--debug", is_flag=True, help="Enable the Flask reloader and debugger.")
def serve(host: str, port: int, debug: bool) -> None:
    """Run the development web server."""
    from cleartusk.web import create_app

    app = create_app()
    click.secho(f"ClearTusk {__version__} on http://{host}:{port}", fg="green", bold=True)
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
