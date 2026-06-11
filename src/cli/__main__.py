import logging

import typer

from src.cli.beacon import app as beacon_app
from src.cli.mssm import app as mssm_app
from src.cli.ucsdh import app as ucsdh_app
from src.cli.vietnam import app as vietnam_app

logger = logging.getLogger(__name__)
cli = typer.Typer(help="Main entry point for the CLI.")
cli.add_typer(beacon_app, name="beacon")
cli.add_typer(ucsdh_app, name="ucsdh")
cli.add_typer(vietnam_app, name="vietnam")
cli.add_typer(mssm_app, name="mssm")


@cli.callback()
def callback() -> None:
    """Inject the logging configuration into the logging system."""
    import logging.config

    from src.config import conf

    logging.config.dictConfig(conf.logging.model_dump())
