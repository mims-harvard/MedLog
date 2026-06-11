"""CLI entry point for the application."""

import copy
import sys
from logging import LogRecord, captureWarnings
from pathlib import Path
from typing import Any

import click
import rich.logging
from rich import pretty, traceback


class RichHandler(rich.logging.RichHandler):
    """Enhanced Rich logging handler with support for colored formatting and traceback customization.

    This handler extends Rich's built-in handler to provide enhanced logging capabilities.

    Features:
        - Automatic warning capture: Captures and formats Python warnings as log messages
        - Pretty printing: Formats complex Python objects in a readable way
        - Customizable traceback handling: Configure how stack traces are displayed
        - Colored log formatting: Supports Rich markup for colorful, styled output

    Note:
        This handler automatically enables warning capture and pretty printing on initialization.
        Traceback handling can be customized through constructor arguments prefixed with `tracebacks_`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the RichHandler with enhanced functionality.

        Args:
            *args: Positional arguments passed to RichHandler
            **kwargs: Keyword arguments including:
                - markup: Enable Rich markup (defaults to True)
                - tracebacks_*: Customization options for traceback handling
        """
        # Ensure markup is enabled by default
        kwargs.setdefault("markup", True)
        super().__init__(*args, **kwargs)

        # Enable warning capture and pretty printing
        captureWarnings(True)
        pretty.install()

        # Suppress tracebacks for click
        traceback_install_kwargs = {"suppress": [click, str(Path(sys.executable).parent)]}
        self._configure_traceback_kwargs(kwargs, traceback_install_kwargs)

        if self.rich_tracebacks:
            traceback.install(**traceback_install_kwargs)

    def _configure_traceback_kwargs(self, input_kwargs: dict, traceback_kwargs: dict) -> None:
        """Process traceback-related configuration from input kwargs.

        Args:
            input_kwargs: Input keyword arguments
            traceback_kwargs: Dict to store processed traceback configuration
        """
        prefix = "tracebacks_"
        for key, value in input_kwargs.items():
            if key.startswith(prefix):
                key_without_prefix = key[len(prefix) :]
                if key_without_prefix == "suppress":
                    traceback_kwargs[key_without_prefix].extend(value)
                else:
                    traceback_kwargs[key_without_prefix] = value

    def emit(self, record: LogRecord) -> None:
        """Emit a log record with optional Rich formatting.

        Args:
            record: The LogRecord to emit

        Raises:
            TypeError: If rich_format is specified but is not a non-empty list
        """
        if not record.args or not hasattr(record, "rich_format"):
            return super().emit(record)

        updated_record = copy.copy(record)
        format_args = record.rich_format

        if not isinstance(format_args, list) or not format_args:
            raise TypeError("rich_format must be a non-empty list")  # noqa: TRY003

        # Apply Rich color formatting to arguments
        new_args = [f"[{color}]{arg}[/{color}]" for arg, color in zip(record.args, format_args)]
        updated_record.args = tuple(new_args)
        super().emit(updated_record)


__all__ = ["RichHandler", "cli"]
