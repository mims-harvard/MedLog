"""Switzerland pilot analysis commands

Figure generation code adapted from the ETH Zurich Ratschlab ai4icu project
(github.com/ratschlab/projects2026-ai4icu-medlog). All commands read a local
assembled parquet (not included; see README); `fairness` falls back to synthetic
data when the parquet is absent.
"""

import typer

app = typer.Typer(help="Switzerland pilot analysis commands.")


@app.command(name="early-alarms")
def early_alarms() -> None:
    """Generate Figure 2 panels b through d (alarm rates by failure group and admission time)."""
    from src.beacon_analysis import main_early_alarms

    main_early_alarms()


@app.command(name="feature-recency")
def feature_recency() -> None:
    """Generate Figure 2 panels e and f (model prediction vs arterial lactate recency)."""
    from src.beacon_analysis import main_feature_recency

    main_feature_recency()


@app.command(name="human-response")
def human_response() -> None:
    """Generate Figure 2 panels g and h (lab order density and time to lab after alarms)."""
    from src.beacon_analysis import main_human_response

    main_human_response()


@app.command()
def fairness() -> None:
    """Generate Figure 2 panels i through k (sex and age AUROC disparity + CUSUM). Uses synthetic data if no parquet."""
    from src.beacon_analysis import main_fairness

    main_fairness()


@app.command()
def stats() -> None:
    """Print BEACON cohort statistics for Table 1 (Bern row)."""
    from src.beacon_analysis import main_stats

    main_stats()
