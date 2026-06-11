from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, model_validator


class LoggingConfig(BaseModel):
    version: int
    disable_existing_loggers: bool
    formatters: dict[str, Any]
    handlers: dict[str, Any]
    loggers: dict[str, Any]
    root: dict[str, Any]


class UcsdhConfig(BaseModel):
    data_dir: Path
    figures_dir: Path
    heatmap_path: Path
    heatmap_barplot_path: Path
    medlog_data_path: Path
    agreement_matrix_path: Path
    summary_stats_path: Path

    @model_validator(mode="after")
    def resolve_paths(self) -> Self:
        """Join relative paths with base directories."""
        # First resolve figures_dir relative to data_dir
        self.figures_dir = self.data_dir / self.figures_dir

        # Paths joined with data_dir
        data_dir_fields = [
            "medlog_data_path",
            "agreement_matrix_path",
            "summary_stats_path",
        ]
        for field_name in data_dir_fields:
            rel_path = getattr(self, field_name)
            setattr(self, field_name, self.data_dir / rel_path)

        # Paths joined with figures_dir
        for field_name in ("heatmap_path", "heatmap_barplot_path"):
            setattr(self, field_name, self.figures_dir / getattr(self, field_name))

        return self


class BeaconConfig(BaseModel):
    data_dir: Path
    figures_dir: Path
    parquet_path: Path

    @model_validator(mode="after")
    def resolve_paths(self) -> Self:
        """Join relative paths with base directories."""
        self.figures_dir = self.data_dir / self.figures_dir
        # Absolute parquet_path is preserved (Path ignores left operand when right is absolute).
        self.parquet_path = self.data_dir / self.parquet_path
        return self


class MssmConfig(BaseModel):
    data_dir: Path
    figures_dir: Path
    cache_dir: Path
    csv_dir: str = ""

    @model_validator(mode="after")
    def resolve_paths(self) -> Self:
        """Join relative paths with base directories."""
        self.figures_dir = self.data_dir / self.figures_dir
        self.cache_dir = self.data_dir / self.cache_dir
        if self.csv_dir:
            self.csv_dir = str(self.data_dir / self.csv_dir)
        return self


class VietnamConfig(BaseModel):
    data_dir: Path
    figures_dir: Path
    waveforms_dir: Path
    medlog_alerts_dir: Path
    medlog_notes_dir: Path
    alert_records_path: Path
    clinical_notes_path: Path
    cleaned_alert_records_path: Path
    cleaned_clinical_notes_path: Path
    cleaned_endpoints_path: Path
    # Subject ID used for the waveform (a) and beat overlay (i) figures.
    # The baseline windows in vietnam_analysis are specific to this subject.
    waveform_subject: str = "24EIc-003-002"

    @model_validator(mode="after")
    def resolve_paths(self) -> Self:
        """Join relative paths with base directories."""
        # First resolve figures_dir relative to data_dir
        self.figures_dir = self.data_dir / self.figures_dir
        self.waveforms_dir = self.data_dir / self.waveforms_dir
        self.medlog_alerts_dir = self.data_dir / self.medlog_alerts_dir
        self.medlog_notes_dir = self.data_dir / self.medlog_notes_dir

        # Paths joined with data_dir
        data_dir_fields = [
            "alert_records_path",
            "clinical_notes_path",
            "cleaned_alert_records_path",
            "cleaned_clinical_notes_path",
            "cleaned_endpoints_path",
        ]
        for field_name in data_dir_fields:
            rel_path = getattr(self, field_name)
            setattr(self, field_name, self.data_dir / rel_path)

        return self
