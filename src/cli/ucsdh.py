"""San Diego pilot analysis commands."""

import typer

app = typer.Typer(help="San Diego pilot analysis commands.")


@app.command()
def heatmap(
    batch: str | None = typer.Option(
        None,
        "--batch",
        help="Limit analysis to this batch (e.g. 2025-12). Saves outputs to figures/<batch>/<batch>_XXX.",
    ),
) -> None:
    """Generate agreement heatmap from UCSD Health pilot data."""
    from src.ucsdh_analysis import main

    main(batch=batch)
