"""Vietnam pilot CLI commands."""

import typer

app = typer.Typer(help="Vietnam pilot analysis commands.")


@app.command()
def prepare() -> None:
    """Clean raw alert and notes data into cleaned CSVs and per alert MedLog JSON records."""
    from src.vietnam_analysis import main_prepare

    main_prepare()


@app.command()
def figures() -> None:
    """Generate Figure 3 panels b through h (trajectories, model probability ECDFs, reason and response bars)."""
    from src.vietnam_analysis import main_figures

    main_figures()


@app.command()
def waveforms(
    subject: str | None = typer.Option(
        None, "--subject", help="Waveform subject ID (default: conf.vietnam.waveform_subject)."
    ),
) -> None:
    """Generate Figure 3 panel a (raw PPG waveforms by alert cohort)."""
    from src.vietnam_analysis import main_waveforms

    main_waveforms(subject=subject)


@app.command(name="beat-overlay")
def beat_overlay(
    subject: str | None = typer.Option(
        None, "--subject", help="Waveform subject ID (default: conf.vietnam.waveform_subject)."
    ),
) -> None:
    """Generate Figure 3 panel i (median beat overlay, dismissed alert vs baseline)."""
    from src.vietnam_analysis import main_beat_overlay

    main_beat_overlay(subject=subject)


@app.command()
def stats() -> None:
    """Print Vietnam cohort statistics for Table 1 (n=15 patients, follow up durations)."""
    from src.vietnam_analysis import main_stats

    main_stats()
