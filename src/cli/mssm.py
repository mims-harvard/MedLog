"""New York pilot analysis commands."""

import typer

app = typer.Typer(help="New York pilot analysis commands.")


@app.command()
def figures() -> None:
    """Render Figure 5 panels a through f from cached aggregates."""
    from src.mssm_analysis import main_figures

    main_figures()


@app.command()
def stats() -> None:
    """Print MSSM reported statistics (calibration gaps + weather deltas) for Figure 5."""
    from src.mssm_analysis import main_stats

    main_stats()
