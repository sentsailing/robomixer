"""Robomixer CLI — import songs, analyze, find transitions."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from robomixer.analysis.pipeline import SUPPORTED_EXTENSIONS, AnalysisPipeline
from robomixer.analysis.transition_points import extract_entry_points, extract_exit_points
from robomixer.cli.metadata import read_metadata
from robomixer.config import settings
from robomixer.models.song import Song
from robomixer.scoring.heuristic import score_transition as heuristic_score
from robomixer.storage.db import Database
from robomixer.storage.features import FeatureStore

logger = logging.getLogger(__name__)

app = typer.Typer(name="robomixer", help="AI-powered DJ transition detection and mixing.")
console = Console()
db = Database()
feature_store = FeatureStore()

# Lazily initialized — only loaded when analysis is actually needed
_pipeline: AnalysisPipeline | None = None


def _get_pipeline() -> AnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AnalysisPipeline()
    return _pipeline


def _run_analysis(song: Song) -> None:
    """Run the full analysis pipeline for a song and store all results."""
    pipeline = _get_pipeline()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running analysis pipeline...", total=None)

        analysis = pipeline.analyze(song)
        progress.update(task, description="Storing analysis results...")

        db.insert_analysis(analysis)

        progress.update(task, description="Extracting transition points...")
        exit_points = extract_exit_points(analysis)
        entry_points = extract_entry_points(analysis)

        progress.update(task, description="Scoring transitions against library...")
        _score_against_library(analysis, exit_points, entry_points)

        progress.update(task, description="Done.")

    console.print(
        f"  BPM: [cyan]{analysis.bpm:.1f}[/cyan]  "
        f"Key: [cyan]{analysis.key}[/cyan] ({analysis.camelot_code})  "
        f"Energy: [cyan]{analysis.average_energy:.3f}[/cyan]  "
        f"Exit points: [cyan]{len(exit_points)}[/cyan]  "
        f"Entry points: [cyan]{len(entry_points)}[/cyan]  "
        f"Vocal regions: [cyan]{len(analysis.vocal_regions)}[/cyan]"
    )


def _score_against_library(analysis, exit_points, entry_points) -> None:
    """Score transitions between this song and all other analyzed songs in the library."""
    all_analyses = db.list_analyses()

    for other in all_analyses:
        if other.song_id == analysis.song_id:
            continue

        other_entry_points = extract_entry_points(other)
        other_exit_points = extract_exit_points(other)

        # Score: this song -> other song (exit from this, enter other)
        for ep in exit_points:
            for np_ in other_entry_points:
                score = heuristic_score(analysis, other, ep, np_)
                db.insert_transition_score(score)

        # Score: other song -> this song (exit from other, enter this)
        for ep in other_exit_points:
            for np_ in entry_points:
                score = heuristic_score(other, analysis, ep, np_)
                db.insert_transition_score(score)


def _import_single(path: Path) -> Song | None:
    """Import a single audio file: read metadata, create Song record, insert into DB.

    Returns the Song on success, None on failure.
    """
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print(
            f"  [yellow]Skipping {path.name}: unsupported format '{path.suffix}'[/yellow]"
        )
        return None

    try:
        meta = read_metadata(path)
    except Exception as e:
        console.print(f"  [red]Failed to read metadata from {path.name}: {e}[/red]")
        return None

    song = Song(
        file_path=path.resolve(),
        title=meta.title,
        artist=meta.artist,
        duration_sec=meta.duration_sec,
        sample_rate=meta.sample_rate,
    )
    db.insert_song(song)
    return song


@app.command()
def import_song(
    path: Path = typer.Argument(..., help="Path to audio file to import"),
    skip_analysis: bool = typer.Option(False, "--skip-analysis", help="Import without running analysis"),
) -> None:
    """Import a song into the library and run analysis."""
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    console.print(f"Importing [bold]{path.name}[/bold]...")
    song = _import_single(path)
    if song is None:
        raise typer.Exit(1)

    minutes = int(song.duration_sec // 60)
    seconds = int(song.duration_sec % 60)
    console.print(
        f"[green]Imported:[/green] {song.title} — {song.artist or 'Unknown Artist'} "
        f"({minutes}:{seconds:02d}) [dim][{song.song_id}][/dim]"
    )

    if not skip_analysis:
        _run_analysis(song)
        console.print("[green]Import and analysis complete.[/green]")


@app.command()
def import_dir(
    directory: Path = typer.Argument(..., help="Directory of audio files to import"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Scan subdirectories"),
    skip_analysis: bool = typer.Option(False, "--skip-analysis", help="Import without running analysis"),
) -> None:
    """Batch-import all audio files from a directory."""
    if not directory.is_dir():
        console.print(f"[red]Not a directory: {directory}[/red]")
        raise typer.Exit(1)

    pattern = "**/*" if recursive else "*"
    audio_files = sorted(
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not audio_files:
        console.print(f"[yellow]No supported audio files found in {directory}[/yellow]")
        return

    console.print(f"Found [bold]{len(audio_files)}[/bold] audio files to import.")
    imported = 0

    for path in audio_files:
        console.print(f"\n  Importing {path.name}...", end=" ")
        song = _import_single(path)
        if song:
            console.print(f"[green]OK[/green] — {song.title}")
            imported += 1
            if not skip_analysis:
                try:
                    _run_analysis(song)
                except Exception as e:
                    console.print(f"  [red]Analysis failed: {e}[/red]")

    console.print(f"\n[green]Imported {imported}/{len(audio_files)} files.[/green]")


@app.command()
def analyze(song_id: str = typer.Argument(..., help="Song UUID to analyze")) -> None:
    """Run or re-run analysis on a song."""
    try:
        uid = UUID(song_id)
    except ValueError:
        console.print(f"[red]Invalid UUID: {song_id}[/red]")
        raise typer.Exit(1)

    song = db.get_song(uid)
    if song is None:
        console.print(f"[red]Song not found: {song_id}[/red]")
        raise typer.Exit(1)

    if not song.file_path.exists():
        console.print(f"[red]Audio file missing: {song.file_path}[/red]")
        raise typer.Exit(1)

    console.print(f"Analyzing [bold]{song.title or song.file_path.name}[/bold]...")
    _run_analysis(song)
    console.print("[green]Analysis complete.[/green]")


@app.command()
def find_transitions(
    song_id: str = typer.Argument(..., help="Song UUID to find transitions from"),
    limit: int = typer.Option(10, help="Max number of transitions to show"),
) -> None:
    """Find the best transitions from a song to other songs in the library."""
    try:
        uid = UUID(song_id)
    except ValueError:
        console.print(f"[red]Invalid UUID: {song_id}[/red]")
        raise typer.Exit(1)

    song = db.get_song(uid)
    if song is None:
        console.print(f"[red]Song not found: {song_id}[/red]")
        raise typer.Exit(1)

    console.print(f"Finding transitions from [bold]{song.title or song.file_path.name}[/bold]...")

    scores = db.get_best_transitions(uid, limit=limit)
    if not scores:
        console.print("[yellow]No transition scores found. Run analysis first.[/yellow]")
        return

    table = Table(title=f"Best Transitions from {song.title}")
    table.add_column("To Song", max_width=20)
    table.add_column("Score", justify="right")
    table.add_column("Harmonic", justify="right")
    table.add_column("Tempo", justify="right")
    table.add_column("Energy", justify="right")
    table.add_column("Technique")
    table.add_column("Exit @", justify="right")
    table.add_column("Entry @", justify="right")

    for s in scores:
        entry_song = db.get_song(s.entry_song_id)
        entry_name = entry_song.title if entry_song else str(s.entry_song_id)[:12]
        table.add_row(
            entry_name,
            f"{s.overall_score:.2f}",
            f"{s.harmonic_score:.2f}",
            f"{s.tempo_score:.2f}",
            f"{s.energy_score:.2f}",
            s.suggested_technique,
            f"{s.exit_timestamp:.1f}s",
            f"{s.entry_timestamp:.1f}s",
        )

    console.print(table)


@app.command()
def library() -> None:
    """List all songs in the library."""
    songs = db.list_songs()
    if not songs:
        console.print("[yellow]Library is empty. Import songs with `robomixer import-song`.[/yellow]")
        return

    table = Table(title="Song Library")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Title")
    table.add_column("Artist")
    table.add_column("Duration", justify="right")
    table.add_column("BPM", justify="right")
    table.add_column("Key")

    for song in songs:
        minutes = int(song.duration_sec // 60)
        seconds = int(song.duration_sec % 60)

        # Try to fetch analysis for BPM/key
        analysis = db.get_analysis(song.song_id)
        bpm_str = f"{analysis.bpm:.1f}" if analysis and analysis.bpm else "-"
        key_str = analysis.camelot_code if analysis and analysis.camelot_code else "-"

        table.add_row(
            str(song.song_id)[:12],
            song.title or song.file_path.stem,
            song.artist or "-",
            f"{minutes}:{seconds:02d}",
            bpm_str,
            key_str,
        )

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, help="API host"),
    port: int = typer.Option(settings.api_port, help="API port"),
) -> None:
    """Start the Robomixer API server."""
    import uvicorn

    console.print(f"Starting Robomixer API at [bold]http://{host}:{port}[/bold]")
    uvicorn.run("robomixer.api.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
