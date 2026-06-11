"""Vietnam pilot analysis.

Entry points:

* :func:`main_prepare` cleans raw alert and notes data.
* :func:`main_figures` generates the Figure 3 panels.
* :func:`main_waveforms` generates panel a.
* :func:`main_beat_overlay` generates panel i.
* :func:`main_stats` prints the Table 1 Vietnam row.
"""

import json
import logging
import uuid
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.signal import butter, filtfilt, find_peaks, resample
from scipy.spatial.distance import pdist, squareform

from src.config import conf

logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)

# Paper figure palette.
C_BLUE = "#507DBC"
C_BLUE_DARK = "#2E4A6F"
C_ACCENT = "#C1666B"

# Ablett phase colours, shared with the trajectory and heatmap styling.
ABLETT_HEATMAP = {
    "before_23": "#81b29a",
    "between_23_34": "#f2cc8f",
    "after_34": "#e07a5f",
}

# Per alert Ablett phase keys and their paper facing titles.
PHASE_SPEC = [
    ("before_23", "Ablett 2"),
    ("between_23_34", "Ablett 3"),
    ("after_34", "Ablett 4"),
]

# Value counts use internal keys; bar labels and colours follow paper order.
REASON_ORDER_LABELED = [
    "Tetanus progression",
    "False alert",
    "Recovery phase",
    "Hospital-acquired infection",
]
REASON_BAR_LABELS = [
    "True disease\nprogression",
    "Unexplained\nalert",
    "Recovery",
    "Hospital-acquired\ninfection",
]

# Waveform constants, shared by both waveform figures.
FS = 100
WINDOW_ROWS = 30
BEAT_RESAMPLE = 80

# Six hand picked alert free PPG windows. These are specific to the default subject
# (conf.vietnam.waveform_subject, "24EIc-003-002"), curated by inspecting that
# subject's recording days, and are not meaningful for any other subject.
VIETNAM_BASELINE_PERIODS = [
    ("2025-03-05 10:15", "2025-03-05 11:30"),
    ("2025-03-12 08:45", "2025-03-12 12:50"),
    ("2025-03-13 08:55", "2025-03-13 13:00"),
    ("2025-03-14 08:35", "2025-03-14 11:50"),
    ("2025-03-18 08:55", "2025-03-18 11:05"),
    ("2025-03-19 08:45", "2025-03-19 11:00"),
]


def _use_arial_if_available() -> None:
    """Set Arial as the matplotlib font when present; otherwise leave the default."""
    if any("Arial" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Arial"


def _save_fig(path: Path) -> None:
    """Save the current figure as PDF + SVG + PNG (600 dpi) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 1. Record cleaning
# ---------------------------------------------------------------------------


def _clean_alert_records() -> pd.DataFrame:
    """Load and clean the raw alert records."""
    alerts = pd.read_csv(
        conf.vietnam.alert_records_path,
        sep=";",
        dtype="string",
        na_values=["NA", "Na", "na", "", "<NA>"],
        keep_default_na=True,
        encoding_errors="replace",
    ).dropna(how="all")

    alerts.columns = alerts.columns.str.strip().str.lower().str.replace(" ", "_")
    alerts = alerts.apply(lambda x: x.str.strip() if x.dtype == "string" else x)

    alerts["alert_datetime"] = pd.to_datetime(
        alerts["date_alert"] + " " + alerts["time_alert"], dayfirst=False, errors="coerce"
    )
    alerts["assessment_datetime"] = pd.to_datetime(
        alerts["date_assessment"] + " " + alerts["time_assessment"], errors="coerce"
    )

    alerts = alerts.assign(
        patient_id=lambda df: df["studyid"].str.replace("24EIc-", "", regex=False),
        model_probability=lambda df: df["model_output"].str.extract(r"(\d+\.\d+)%").astype(float),
        response_time_min=lambda df: (df["assessment_datetime"] - df["alert_datetime"]).dt.total_seconds() / 60,
        hour=lambda df: df["alert_datetime"].dt.hour,
        day_of_week=lambda df: df["alert_datetime"].dt.dayofweek,
        date=lambda df: df["alert_datetime"].dt.date,
    ).drop(columns=["studyid"])

    mask_negative = alerts["response_time_min"] < 0
    num_negatives = int(mask_negative.sum())
    if num_negatives > 0:
        logger.warning(
            "%d records have negative response times (range: %.2f to %.2f min).",
            num_negatives,
            alerts["response_time_min"].min(),
            alerts.loc[mask_negative, "response_time_min"].max(),
        )

    alerts["response_time_min"] = alerts["response_time_min"].clip(lower=0)

    cols_order = [
        "patient_id",
        "alert_datetime",
        "date",
        "hour",
        "day_of_week",
        "assessment_datetime",
        "response_time_min",
        "current_ablett",
        "model_output",
        "model_probability",
        "possible_reason",
        "event_at_alert",
        "note",
    ]
    alerts = alerts[[c for c in cols_order if c in alerts.columns]]

    if alerts["model_output"].isna().any():
        num_output_na = int(alerts["model_output"].isna().sum())
        logger.info("Removing %d alerts that have NA model output.", num_output_na)
        alerts = alerts[alerts["model_output"].notna()]

    if alerts["model_probability"].isna().any():
        num_prob_na = int(alerts["model_probability"].isna().sum())
        logger.info("Removing %d alerts that have NA model probability.", num_prob_na)
        alerts = alerts[alerts["model_probability"].notna()]

    logger.info(
        "Loaded %d records for %d patients. All negatives clipped to 0.",
        len(alerts),
        alerts["patient_id"].nunique(),
    )
    logger.info("Number of records with non-empty alert reason: %d", int(alerts["possible_reason"].notna().sum()))
    return alerts


def _create_medlog_record(row: pd.Series) -> tuple[dict, str]:
    """Build one MedLog JSON record (and its event_id) for a single alert row."""
    event_id = str(uuid.uuid4())
    record = {
        "header": {
            "medlog_version": "0.0.1",
            "event_id": event_id,
            "timestamp": row["alert_datetime"].isoformat() if pd.notna(row["alert_datetime"]) else None,
            "system_info": "Vietnam Clinical Alert System",
            "timezone": "ICT",
        },
        "model_instance": {"model_id": "24EIC-AI-Alert", "model_version": "1.0.0"},
        "user_identity": {},
        "target_identity": {"patient_id": row["patient_id"]},
        "inputs": {
            "current_ablett_grade": row.get("current_ablett") if pd.notna(row["current_ablett"]) else "",
            "context": {"day_of_week": row.get("day_of_week"), "hour": row.get("hour")},
        },
        "internal_artifacts": {
            "model_probability": row["model_probability"] / 100.0 if pd.notna(row["model_probability"]) else ""
        },
        "outputs": {
            "model_output": row.get("model_output"),
        },
        "outcomes": {
            "assessment_timestamp": row["assessment_datetime"].isoformat()
            if pd.notna(row["assessment_datetime"])
            else "",
            "response_time_minutes": row.get("response_time_min"),
            "event_at_alert": row.get("event_at_alert"),
            "clinical_note": row.get("note") if pd.notna(row["note"]) else "",
        },
        "user_feedback": {
            "clinician_assessment": row.get("possible_reason"),
        },
    }
    return record, event_id


def _write_medlog_records(alerts: pd.DataFrame) -> pd.DataFrame:
    """Write one MedLog JSON per alert and save the cleaned alert records CSV.

    The MedLog alerts directory is cleared (existing files unlinked) before writing.
    Returns the alert frame with an ``eid`` column prepended.
    """
    output_dir = conf.vietnam.medlog_alerts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir = conf.vietnam.cleaned_alert_records_path.parent
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    # Destructive clear, scoped strictly to the MedLog alerts directory.
    for file in output_dir.iterdir():
        if file.is_file():
            file.unlink()

    eids: list[str] = []
    for _, row in alerts.iterrows():
        medlog_json, eid = _create_medlog_record(row)
        eids.append(eid)
        filename = f"medlog_{row['patient_id']}_{eid[:8]}.json"
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(medlog_json, f, indent=4, default=str)

    alerts = alerts.copy()
    alerts["eid"] = eids
    alerts = alerts[["eid"] + [c for c in alerts.columns if c != "eid"]]
    alerts.to_csv(conf.vietnam.cleaned_alert_records_path, index=False)
    logger.info("Successfully exported %d MedLog records to %s", len(alerts), output_dir)
    return alerts


def _clean_clinical_notes() -> pd.DataFrame:
    """Load and clean the clinical annotations sheet (note_output_cleaned_LT)."""
    clinical_raw = pd.read_excel(conf.vietnam.clinical_notes_path, sheet_name="note_output_cleaned_LT")

    all_na_rows = clinical_raw.isna().all(axis=1)
    logger.info("Dropping %d rows with all NA values", int(all_na_rows.sum()))
    clinical_raw = clinical_raw[~all_na_rows]

    mapping = {
        "note": "vietnam_clinical_note",
        "Louise note": "posthoc_clinical_note",
        "Alert Code": "alert_code",
        "CORRECTED START": "event_datetime",
        "Alert correctly identifying patient event": "alert_correct_patient_event",
        "Alert correctly identifying worsening in Tetanus severity or Ablett": "alert_correct_worsening_severity",
        "initial": "vietnam_reviewer",
        "yob": "patient_yob",
        "gender": "patient_gender",
    }

    clinical = clinical_raw.rename(columns=mapping)
    clinical.columns = clinical.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    clinical = clinical.apply(lambda x: x.str.strip() if isinstance(x, str) else x)
    clinical["patient_id"] = clinical["studyid"].str.replace("24EIc-", "", regex=False)

    clinical["event_datetime"] = pd.to_datetime(clinical["event_datetime"], errors="coerce")
    clinical["date"] = clinical["event_datetime"].dt.date
    clinical["hour"] = clinical["event_datetime"].dt.hour
    clinical["day_of_week"] = clinical["event_datetime"].dt.dayofweek

    cols_order = [
        "patient_id",
        "patient_yob",
        "patient_gender",
        "event_datetime",
        "date",
        "hour",
        "day_of_week",
        "alert_code",
        "alert_correct_patient_event",
        "alert_correct_worsening_severity",
        "vietnam_clinical_note",
        "vietnam_reviewer",
        "posthoc_clinical_note",
    ]
    clinical = clinical[[c for c in cols_order if c in clinical.columns]]
    clinical.to_csv(conf.vietnam.cleaned_clinical_notes_path, index=False)
    logger.info("Cleaned %d clinical events.", len(clinical))
    return clinical


def _clean_endpoints() -> pd.DataFrame:
    """Load and clean the Main endpoints sheet (enrollment + Ablett transitions)."""
    endpoints_raw = pd.read_excel(conf.vietnam.clinical_notes_path, sheet_name="Main endpoints")

    endpoints = endpoints_raw.copy()
    endpoints.columns = (
        endpoints.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("__", "_", regex=False)
    )

    mapping = {
        "unnamed:_7": "ablett_3_4_transition_time",
        "ablet_3_4_transition": "ablett_3_4_transition_date",
        "ablett_at_enrolment": "enrollment_ablett",
        "date_enrolment": "date_enrollment",
        "time_enrolment": "time_enrollment",
        "significant_event_1": "significant_event1",
    }
    endpoints = endpoints.rename(columns=mapping)
    endpoints = endpoints.apply(lambda x: x.str.strip() if isinstance(x, str) else x)
    endpoints["patient_id"] = endpoints["studyid"].str.replace("24EIc-", "", regex=False)

    # Handle the '?2' value found in row 14 for enrollment ablett
    endpoints["enrollment_ablett"] = (
        endpoints["enrollment_ablett"]
        .astype(str)
        .str.replace("?", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )

    def combine_dt(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
        if date_col not in df.columns or time_col not in df.columns:
            return pd.Series([pd.NaT] * len(df))
        d = df[date_col].astype(str).str.replace(r"\.0$", "", regex=True).replace(["nan", "nat", "none"], np.nan)
        t = df[time_col].astype(str).str.replace(r"\.0$", "", regex=True).replace(["nan", "nat", "none"], np.nan)
        combined = d + " " + t
        return pd.to_datetime(combined, errors="coerce", format="mixed")

    endpoints["enrollment_datetime"] = combine_dt(endpoints, "date_enrollment", "time_enrollment")
    endpoints["ablett_2_3_transition_datetime"] = combine_dt(
        endpoints, "ablett_2_3_transition_date", "ablett_2_3_transition_time"
    )
    endpoints["ablett_3_4_transition_datetime"] = combine_dt(
        endpoints, "ablett_3_4_transition_date", "ablett_3_4_transition_time"
    )
    endpoints["significant_event_1_datetime"] = combine_dt(
        endpoints, "date_significant_event1", "time_significant_event_1"
    )
    endpoints["significant_event_2_datetime"] = combine_dt(
        endpoints, "date_significant_event2", "time_significant_event_2"
    )

    cols_order = [
        "patient_id",
        "enrollment_datetime",
        "enrollment_ablett",
        "ablett_2_3_transition_datetime",
        "ablett_3_4_transition_datetime",
        "significant_event_1",
        "significant_event_1_datetime",
        "significant_event_2",
        "significant_event_2_datetime",
        "patient_summary",
    ]
    endpoints = endpoints[[c for c in cols_order if c in endpoints.columns]]
    endpoints.to_csv(conf.vietnam.cleaned_endpoints_path, index=False)
    logger.info("Processed %d patient endpoints.", len(endpoints))
    return endpoints


def main_prepare() -> None:
    """Clean raw alert/notes data into cleaned CSVs + per-alert MedLog JSON records."""
    _use_arial_if_available()

    conf.vietnam.figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Alert records path: %s", conf.vietnam.alert_records_path)
    logger.info("Clinical notes path: %s", conf.vietnam.clinical_notes_path)
    logger.info("Figures output path: %s", conf.vietnam.figures_dir)
    logger.info("MedLog alerts path: %s", conf.vietnam.medlog_alerts_dir)

    if not conf.vietnam.alert_records_path.exists():
        msg = f"Alert records file not found: {conf.vietnam.alert_records_path}"
        raise FileNotFoundError(msg)
    if not conf.vietnam.clinical_notes_path.exists():
        msg = f"Clinical notes file not found: {conf.vietnam.clinical_notes_path}"
        raise FileNotFoundError(msg)

    alerts = _clean_alert_records()
    _write_medlog_records(alerts)
    _clean_clinical_notes()
    _clean_endpoints()


# ---------------------------------------------------------------------------
# 2. Paper figures
# ---------------------------------------------------------------------------


def _require_cleaned(path: Path, name: str) -> None:
    """Raise a helpful error if a cleaned CSV is missing."""
    if not path.exists():
        msg = f"Cleaned {name} not found: {path}. Run `uv run cli vietnam prepare` first."
        raise FileNotFoundError(msg)


def _norm(s) -> object:
    """Collapse whitespace in a free-text field, preserving NaN."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    return " ".join(s.split())


def _event_status(s) -> object:
    """Map ``event_at_alert`` free text to Handled or Ignored."""
    if pd.isna(s):
        return np.nan
    t = str(s).strip().lower()
    if t.startswith("alert ignored"):
        return "Ignored"
    if t.startswith("alert handled"):
        return "Handled"
    return np.nan


def _load_alerts_with_phase() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned alerts and endpoints and derive per alert Ablett phase, reason, and status.

    Returns ``(alerts_with_trans, endpoints)``. The Ablett phase (one of before_23,
    between_23_34, after_34) is computed by merging each alert against its patient's
    transition times.
    """
    _require_cleaned(conf.vietnam.cleaned_alert_records_path, "alert records")
    _require_cleaned(conf.vietnam.cleaned_endpoints_path, "endpoints")

    alerts = pd.read_csv(
        conf.vietnam.cleaned_alert_records_path,
        parse_dates=["alert_datetime", "assessment_datetime"],
    )
    for col in ("hour", "day_of_week"):
        if col in alerts.columns:
            alerts[col] = pd.to_numeric(alerts[col], errors="coerce")

    endpoints = pd.read_csv(
        conf.vietnam.cleaned_endpoints_path,
        parse_dates=[
            "enrollment_datetime",
            "ablett_2_3_transition_datetime",
            "ablett_3_4_transition_datetime",
            "significant_event_1_datetime",
            "significant_event_2_datetime",
        ],
    )

    alerts["time_of_day"] = np.where(alerts["hour"].between(7, 18), "Day (07:00-19:00)", "Night (19:00-07:00)")
    alerts["day_type"] = np.where(alerts["day_of_week"] >= 5, "Weekend", "Weekday")

    alerts["possible_reason_norm"] = alerts.get("possible_reason", pd.Series(index=alerts.index, dtype="string")).map(
        _norm
    )

    reason_map = {
        "False Alert": "False alert",
        "No Idea": "False alert",
        "Change Ablett score": "Tetanus progression",
        "Noised data, weaning time, HAI": "Hospital-acquired infection",
        "Weaning and HAI": "Hospital-acquired infection",
        "Weaning time": "Recovery phase",
        "Patient change position": "Recovery phase",
        "Sensor error due to long pressure": "Recovery phase",
        "Noised data, weaning time": "Recovery phase",
    }
    alerts["reason_category"] = alerts["possible_reason_norm"].map(reason_map)
    alerts.loc[alerts["possible_reason_norm"].isna(), "reason_category"] = "Not labeled"

    alerts["event_status"] = alerts.get("event_at_alert", pd.Series(index=alerts.index, dtype="string")).map(
        _event_status
    )

    transition_cols = [
        "patient_id",
        "ablett_2_3_transition_datetime",
        "ablett_3_4_transition_datetime",
    ]
    alerts_with_trans = alerts.merge(endpoints[transition_cols], on="patient_id", how="left")

    t = alerts_with_trans["alert_datetime"]
    t23 = alerts_with_trans["ablett_2_3_transition_datetime"]
    t34 = alerts_with_trans["ablett_3_4_transition_datetime"]

    phase = pd.Series(pd.NA, index=alerts_with_trans.index, dtype="string")
    valid = t.notna()
    neither_transition = t23.isna() & t34.isna()
    before_neither = valid & neither_transition
    before_before_t23 = valid & ~neither_transition & t23.notna() & (t < t23)
    before = before_neither | before_before_t23
    after = valid & t34.notna() & (t >= t34)
    between = valid & ~before & ~after

    phase.loc[before] = "before_23"
    phase.loc[after] = "after_34"
    phase.loc[between] = "between_23_34"
    alerts_with_trans["ablett_phase"] = phase

    return alerts_with_trans, endpoints


def _annotate_bar_counts(ax, bars, counts) -> None:
    for bar, n in zip(bars, counts):
        h = bar.get_height()
        y = h + max(ax.get_ylim()[1] * 0.02, 1.0) if h > 0 else 1.0
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{int(n)}", ha="center", va="bottom", fontsize=10)


def _annotate_bar_props(ax, bars, values) -> None:
    for bar, h in zip(bars, values):
        if h > 0:
            y = h + max(ax.get_ylim()[1] * 0.01, 0.5)
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)


def _reason_counts_matrix(work: pd.DataFrame) -> np.ndarray:
    """Rows = reason order, cols = Ablett phase (before_23, between_23_34, after_34)."""
    mat = np.zeros((len(REASON_ORDER_LABELED), len(PHASE_SPEC)), dtype=float)
    for j, (phase_key, _) in enumerate(PHASE_SPEC):
        sub = work[work["ablett_phase"] == phase_key]
        vc = sub["reason_category"].value_counts()
        for i, r in enumerate(REASON_ORDER_LABELED):
            mat[i, j] = int(vc.get(r, 0))
    return mat


def _prepare_alert_reasons(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Shared data prep for alert reason plots (hours 6 to 16, labeled only)."""
    work = df[(df["hour"] >= 6) & (df["hour"] < 16)].copy()
    work = work[work["reason_category"] != "Not labeled"].copy()
    counts = _reason_counts_matrix(work)
    col_totals = counts.sum(axis=0, keepdims=True)
    col_totals[col_totals == 0] = 1
    props = counts / col_totals * 100

    legend_labels = []
    for phase_key, phase_title in PHASE_SPEC:
        sub = work[work["ablett_phase"] == phase_key]
        n_pat = sub["patient_id"].nunique()
        n_rec = len(sub)
        legend_labels.append(f"{phase_title} ({n_pat} patients, {n_rec} MedLog records)")
    return counts, props, legend_labels


def plot_alert_reasons_counts(df: pd.DataFrame, save_path: Path) -> None:
    """Alert reasons: raw counts by Ablett group (panel g)."""
    counts, _, legend_labels = _prepare_alert_reasons(df)
    n_reasons = len(REASON_ORDER_LABELED)
    x = np.arange(n_reasons, dtype=float)
    n_phase = len(PHASE_SPEC)
    width = 0.22
    offsets = (np.arange(n_phase) - (n_phase - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (phase_key, _) in enumerate(PHASE_SPEC):
        bars = ax.bar(
            x + offsets[j],
            counts[:, j],
            width,
            color=ABLETT_HEATMAP[phase_key],
            edgecolor="black",
            linewidth=0.6,
            label=legend_labels[j],
            zorder=4,
        )
        _annotate_bar_counts(ax, bars, counts[:, j])

    ax.set_xticks(x)
    ax.set_xticklabels(REASON_BAR_LABELS, fontsize=9)
    ax.set_ylabel("Number of alerts")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)
    fig.subplots_adjust(bottom=0.22)
    _save_fig(save_path)


def plot_alert_reasons_prop(df: pd.DataFrame, save_path: Path) -> None:
    """Alert reasons: proportion within each Ablett group (panel h)."""
    _, props, legend_labels = _prepare_alert_reasons(df)
    n_reasons = len(REASON_ORDER_LABELED)
    x = np.arange(n_reasons, dtype=float)
    n_phase = len(PHASE_SPEC)
    width = 0.22
    offsets = (np.arange(n_phase) - (n_phase - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (phase_key, _) in enumerate(PHASE_SPEC):
        bars = ax.bar(
            x + offsets[j],
            props[:, j],
            width,
            color=ABLETT_HEATMAP[phase_key],
            edgecolor="black",
            linewidth=0.6,
            label=legend_labels[j],
            zorder=4,
        )
        _annotate_bar_props(ax, bars, props[:, j])

    ax.set_xticks(x)
    ax.set_xticklabels(REASON_BAR_LABELS, fontsize=9)
    ax.set_ylabel("Proportion of alerts within Ablett group (%)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)
    fig.subplots_adjust(bottom=0.22)
    _save_fig(save_path)


def _prepare_clinician_response(df: pd.DataFrame) -> tuple[list[str], list[tuple], list[str]]:
    """Shared data prep for clinician response plots (hours 6 to 16)."""
    cats = ["Handled", "Ignored"]
    work = df[(df["hour"] >= 6) & (df["hour"] < 16)].copy()

    phase_data = []
    legend_labels = []
    for phase_key, phase_title in PHASE_SPEC:
        sub = work[work["ablett_phase"] == phase_key]
        n_pat = sub["patient_id"].nunique()
        n_rec = len(sub)
        vc = sub["event_status"].value_counts()
        cnts = np.array([int(vc.get(c, 0)) for c in cats], dtype=float)
        total = cnts.sum()
        prps = (cnts / total * 100) if total > 0 else np.zeros_like(cnts)
        phase_data.append((phase_key, phase_title, n_pat, n_rec, cnts, prps))
        legend_labels.append(f"{phase_title} ({n_pat} patients, {n_rec} MedLog records)")
    return cats, phase_data, legend_labels


def plot_clinician_response_counts(df: pd.DataFrame, save_path: Path) -> None:
    """Clinician response: raw counts by Ablett group (panel e)."""
    cats, phase_data, legend_labels = _prepare_clinician_response(df)
    n_cats = len(cats)
    x = np.arange(n_cats, dtype=float)
    n_phase = len(PHASE_SPEC)
    width = 0.22
    offsets = (np.arange(n_phase) - (n_phase - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(6, 4))
    for j, (phase_key, _, _, _, cnts, _) in enumerate(phase_data):
        bars = ax.bar(
            x + offsets[j],
            cnts,
            width,
            color=ABLETT_HEATMAP[phase_key],
            edgecolor="black",
            linewidth=0.6,
            label=legend_labels[j],
            zorder=4,
        )
        _annotate_bar_counts(ax, bars, cnts)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("Number of alerts")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.22)
    _save_fig(save_path)


def plot_clinician_response_prop(df: pd.DataFrame, save_path: Path) -> None:
    """Clinician response: proportion within each Ablett group (panel f)."""
    cats, phase_data, legend_labels = _prepare_clinician_response(df)
    n_cats = len(cats)
    x = np.arange(n_cats, dtype=float)
    n_phase = len(PHASE_SPEC)
    width = 0.22
    offsets = (np.arange(n_phase) - (n_phase - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(6, 4))
    for j, (phase_key, _, _, _, _, prps) in enumerate(phase_data):
        bars = ax.bar(
            x + offsets[j],
            prps,
            width,
            color=ABLETT_HEATMAP[phase_key],
            edgecolor="black",
            linewidth=0.6,
            label=legend_labels[j],
            zorder=4,
        )
        _annotate_bar_props(ax, bars, prps)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("Proportion of alerts within Ablett group (%)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.22)
    _save_fig(save_path)


# Line plots by hour of day (one panel per Ablett phase).
REASON_ORDER_LINE = [
    "Tetanus progression",
    "False alert",
    "Recovery phase",
    "Hospital-acquired infection",
]
REASON_LINE_STYLE = {
    "Tetanus progression": {"color": "#1B3A5C", "marker": "s"},
    "False alert": {"color": "#A8D4F0", "marker": "o"},
    "Recovery phase": {"color": "#6AABD2", "marker": "^"},
    "Hospital-acquired infection": {"color": "#3670A0", "marker": "D"},
}
REASON_LINE_LABELS = {
    "False alert": "Unexplained alert",
    "Tetanus progression": "True disease progression",
    "Recovery phase": "Recovery",
    "Hospital-acquired infection": "Hospital-acquired infection",
}
STATUS_LINE_STYLE = {
    "Handled": {"color": "#1B3A5C", "marker": "o"},
    "Ignored": {"color": "#6AABD2", "marker": "s"},
}


def plot_alert_reasons_by_hour_ablett(df: pd.DataFrame, save_path: Path) -> None:
    """Line counts per hour (6 to 16) by reason_category, one panel per Ablett phase.

    Drops unlabeled alerts.
    """
    all_hours = range(6, 16)
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    plt.subplots_adjust(bottom=0.28, wspace=0.28)

    for ax, (phase_key, phase_title) in zip(axes, PHASE_SPEC):
        sub = df[df["ablett_phase"] == phase_key]
        sub = sub[sub["reason_category"] != "Not labeled"]
        sub = sub[(sub["hour"] >= 6) & (sub["hour"] < 16)]
        n_pat = sub["patient_id"].nunique()
        n_alr = len(sub)
        hourly = pd.crosstab(sub["hour"], sub["reason_category"])
        hourly = hourly.reindex(index=list(all_hours), columns=REASON_ORDER_LINE, fill_value=0)

        for reason in REASON_ORDER_LINE:
            sty = REASON_LINE_STYLE[reason]
            ax.plot(
                hourly.index,
                hourly[reason],
                marker=sty["marker"],
                linestyle="-",
                label=REASON_LINE_LABELS[reason],
                linewidth=1.8,
                markersize=6,
                color=sty["color"],
            )
        ax.set_title(f"{phase_title} ({n_pat} patients, {n_alr} alerts)")
        ax.set_xlabel("Hour of day", fontsize=11)
        ax.set_ylabel("Number of alerts", fontsize=11)
        ax.set_xticks(list(all_hours))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    handles = [
        Line2D(
            [0], [0], color=REASON_LINE_STYLE[r]["color"], lw=1.8, marker=REASON_LINE_STYLE[r]["marker"], markersize=6
        )
        for r in REASON_ORDER_LINE
    ]
    fig.legend(
        handles,
        [REASON_LINE_LABELS[r] for r in REASON_ORDER_LINE],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=4,
        frameon=False,
    )
    _save_fig(save_path)


def plot_clinician_response_by_hour_ablett(df: pd.DataFrame, save_path: Path) -> None:
    """Line counts per hour (6 to 16) by event_status, one panel per Ablett phase."""
    all_hours = range(6, 16)
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    plt.subplots_adjust(bottom=0.22, wspace=0.28)

    for ax, (phase_key, phase_title) in zip(axes, PHASE_SPEC):
        sub = df[df["ablett_phase"] == phase_key]
        sub = sub[(sub["hour"] >= 6) & (sub["hour"] < 16)]
        n_pat = sub["patient_id"].nunique()
        n_alr = len(sub)
        hourly = pd.crosstab(sub["hour"], sub["event_status"])
        for col in ["Handled", "Ignored"]:
            if col not in hourly.columns:
                hourly[col] = 0
        hourly = hourly[["Handled", "Ignored"]].reindex(index=list(all_hours), fill_value=0)

        for status in ["Handled", "Ignored"]:
            sty = STATUS_LINE_STYLE[status]
            ax.plot(
                hourly.index,
                hourly[status],
                marker=sty["marker"],
                linestyle="-",
                label=status,
                linewidth=1.8,
                markersize=6,
                color=sty["color"],
            )
        ax.set_title(f"{phase_title} ({n_pat} patients, {n_alr} alerts)")
        ax.set_xlabel("Hour of day", fontsize=11)
        ax.set_ylabel("Number of alerts", fontsize=11)
        ax.set_xticks(list(all_hours))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    handles = [
        Line2D(
            [0], [0], color=STATUS_LINE_STYLE[s]["color"], lw=1.8, marker=STATUS_LINE_STYLE[s]["marker"], markersize=6
        )
        for s in ["Handled", "Ignored"]
    ]
    fig.legend(
        handles,
        ["Handled", "Ignored"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        frameon=False,
    )
    _save_fig(save_path)


def _model_prob_support(df: pd.DataFrame, col: str = "model_probability") -> tuple[float, float]:
    s = df[col].dropna()
    if s.empty:
        return 0.0, 100.0
    return float(s.min()), float(s.max())


def plot_model_prob_cumulative(
    df: pd.DataFrame,
    save_path: Path,
    group_a,
    group_b,
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    col: str = "model_probability",
    n_bins: int = 30,
) -> None:
    """Stacked two panel figure: top is the ECDF, bottom is an overlaid density histogram (panels c, d)."""
    x_a = df.loc[group_a, col].dropna().sort_values().to_numpy()
    x_b = df.loc[group_b, col].dropna().sort_values().to_numpy()
    lo, hi = _model_prob_support(df, col)

    _fig, (ax_ecdf, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(3.5, 3.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.2},
    )

    for data, label, color in [(x_a, label_a, color_a), (x_b, label_b, color_b)]:
        if len(data) == 0:
            continue
        y_ecdf = np.arange(1, len(data) + 1) / len(data)
        ax_ecdf.step(data, y_ecdf, where="post", label=f"{label} (n={len(data)})", color=color, linewidth=2)

    ax_ecdf.set_ylabel("eCDF")
    ax_ecdf.set_ylim(0, 1.02)
    ax_ecdf.set_xlim(left=lo, right=hi)
    ax_ecdf.spines["top"].set_visible(False)
    ax_ecdf.spines["right"].set_visible(False)
    ax_ecdf.grid(True, axis="both", linestyle="--", alpha=0.45)

    bin_edges = np.linspace(lo, hi, n_bins + 1)
    for data, label, color in [(x_a, label_a, color_a), (x_b, label_b, color_b)]:
        if len(data) == 0:
            continue
        ax_hist.hist(
            data,
            bins=bin_edges,
            density=True,
            alpha=0.45,
            color=color,
            edgecolor=color,
            linewidth=0.8,
            label=f"{label} (n={len(data)})",
        )

    ax_hist.set_xlabel("Model probability (%)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_xlim(left=lo, right=hi)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.grid(True, axis="y", linestyle="--", alpha=0.45, zorder=0)

    handles, labels = ax_ecdf.get_legend_handles_labels()
    ax_hist.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.8), ncol=1, frameon=False)

    _save_fig(save_path)


def plot_patient_trajectories(df: pd.DataFrame, save_path: Path) -> None:  # noqa: C901
    """Horizontal Ablett grade trajectory bars per patient (panel b, base version)."""

    def get_sort_key(row):
        t0 = row["enrollment_datetime"]
        g0 = row["enrollment_ablett"]

        def rel_h(t):
            if pd.isna(t) or pd.isna(t0):
                return 9999
            return (t - t0).total_seconds() / 3600

        h23 = rel_h(row["ablett_2_3_transition_datetime"])
        h34 = rel_h(row["ablett_3_4_transition_datetime"])

        if g0 == 2 and h23 == 9999:
            return (0, 0, 0)
        elif g0 == 2 and h23 != 9999:
            return (1, h23, h34)
        elif g0 == 3 and h34 == 9999:
            return (2, 0, 0)
        elif g0 == 3 and h34 != 9999:
            return (3, 0, h34)
        return (4, 0, 0)

    df = df.copy()
    df["sort_key"] = df.apply(get_sort_key, axis=1)
    df = df.sort_values("sort_key", ascending=True).reset_index(drop=True)
    df = df.iloc[::-1].reset_index(drop=True)

    n_patients = len(df)
    ablett_colors = {
        2: "#81b29a",
        3: "#f2cc8f",
        4: "#e07a5f",
    }

    _fig, ax = plt.subplots(figsize=(10, 8))
    x_max = 100

    for i, row in df.iterrows():
        t0 = row["enrollment_datetime"]
        g_start = int(row["enrollment_ablett"]) if not pd.isna(row["enrollment_ablett"]) else 2

        def to_hours(t, _t0=t0):
            if pd.isna(t) or pd.isna(_t0):
                return np.nan
            return (t - _t0).total_seconds() / 3600

        h23 = to_hours(row["ablett_2_3_transition_datetime"])
        h34 = to_hours(row["ablett_3_4_transition_datetime"])

        current_x = 0

        # Grade 2
        if g_start == 2:
            end_2 = min(h23, x_max) if not pd.isna(h23) else x_max
            ax.hlines(y=i, xmin=0, xmax=end_2, colors=ablett_colors[2], linewidth=10, capstyle="butt")
            current_x = end_2
            if not pd.isna(h23) and h23 <= x_max:
                ax.vlines(x=h23, ymin=i - 0.4, ymax=i + 0.4, color="black", lw=1.5, zorder=10)

        # Grade 3
        if (g_start == 3) or (g_start == 2 and not pd.isna(h23) and h23 < x_max):
            start_3 = current_x
            end_3 = min(h34, x_max) if not pd.isna(h34) else x_max
            if end_3 > start_3:
                ax.hlines(y=i, xmin=start_3, xmax=end_3, colors=ablett_colors[3], linewidth=10, capstyle="butt")
                current_x = end_3
                if not pd.isna(h34) and h34 <= x_max:
                    ax.vlines(x=h34, ymin=i - 0.4, ymax=i + 0.4, color="black", lw=1.5, zorder=10)

        # Grade 4
        if current_x < x_max:
            final_g = 4 if (g_start == 4 or not pd.isna(h34)) else (3 if (g_start == 3 or not pd.isna(h23)) else 2)
            ax.hlines(y=i, xmin=current_x, xmax=x_max, colors=ablett_colors[final_g], linewidth=10, capstyle="butt")

    ax.set_yticks(range(n_patients))
    ax.set_yticklabels(df["patient_id"])
    ax.set_xlabel("Hours since study enrollment", size=12)
    ax.set_xlim(0, x_max)

    legend_elements = [
        Line2D([0], [0], color=ablett_colors[2], lw=6, label="Ablett 2"),
        Line2D([0], [0], color=ablett_colors[3], lw=6, label="Ablett 3"),
        Line2D([0], [0], color=ablett_colors[4], lw=6, label="Ablett 4"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False)

    sns.despine()
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    _save_fig(save_path)


def plot_patient_trajectories_with_alerts(  # noqa: C901
    endpoints_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    save_path: Path,
    show_segment_dividers: bool = False,
) -> None:
    """Patient trajectories with per patient alert KDE and rug overlay (panel b, annotated)."""
    from scipy.stats import gaussian_kde

    def get_sort_key(row):
        t0 = row["enrollment_datetime"]
        g0 = row["enrollment_ablett"]

        def rel_h(t):
            if pd.isna(t) or pd.isna(t0):
                return 9999
            return (t - t0).total_seconds() / 3600

        h23 = rel_h(row["ablett_2_3_transition_datetime"])
        h34 = rel_h(row["ablett_3_4_transition_datetime"])

        if g0 == 2 and h23 == 9999:
            return (0, 0, 0)
        elif g0 == 2 and h23 != 9999:
            return (1, h23, h34)
        elif g0 == 3 and h34 == 9999:
            return (2, 0, 0)
        elif g0 == 3 and h34 != 9999:
            return (3, 0, h34)
        return (4, 0, 0)

    df = endpoints_df.copy()
    df["sort_key"] = df.apply(get_sort_key, axis=1)
    df = df.sort_values("sort_key", ascending=True).reset_index(drop=True)

    alerts_merged = alerts_df.merge(df[["patient_id", "enrollment_datetime"]], on="patient_id", how="left")
    alerts_merged["rel_hours"] = (
        alerts_merged["alert_datetime"] - alerts_merged["enrollment_datetime"]
    ).dt.total_seconds() / 3600

    n_patients = len(df)
    x_max = 100
    ablett_colors = {2: "#81b29a", 3: "#f2cc8f", 4: "#e07a5f"}

    height_ratios = [2, 3] * n_patients
    fig, axes = plt.subplots(
        nrows=2 * n_patients,
        ncols=1,
        figsize=(8, 0.4 * n_patients),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.0},
    )

    def short_label(pid):
        parts = str(pid).split("-")
        return str(int(parts[-1]))

    x_grid = np.linspace(0, x_max, 500)

    for i, row in df.iterrows():
        ax_kde = axes[2 * i]
        ax_traj = axes[2 * i + 1]
        pid = row["patient_id"]

        t0 = row["enrollment_datetime"]
        g_start = int(row["enrollment_ablett"]) if not pd.isna(row["enrollment_ablett"]) else 2

        def to_hours(t, _t0=t0):
            if pd.isna(t) or pd.isna(_t0):
                return np.nan
            return (t - _t0).total_seconds() / 3600

        h23 = to_hours(row["ablett_2_3_transition_datetime"])
        h34 = to_hours(row["ablett_3_4_transition_datetime"])

        current_x = 0
        transition_points = []

        if g_start == 2:
            end_2 = min(h23, x_max) if not pd.isna(h23) else x_max
            ax_traj.hlines(y=0.5, xmin=0, xmax=end_2, colors=ablett_colors[2], linewidth=30, capstyle="butt")
            if not pd.isna(h23) and h23 < x_max:
                transition_points.append(end_2)
            current_x = end_2

        if (g_start == 3) or (g_start == 2 and not pd.isna(h23) and h23 < x_max):
            start_3 = current_x
            end_3 = min(h34, x_max) if not pd.isna(h34) else x_max
            if end_3 > start_3:
                ax_traj.hlines(y=0.5, xmin=start_3, xmax=end_3, colors=ablett_colors[3], linewidth=30, capstyle="butt")
                if not pd.isna(h34) and h34 < x_max:
                    transition_points.append(end_3)
                current_x = end_3

        if current_x < x_max:
            final_g = 4 if (g_start == 4 or not pd.isna(h34)) else (3 if (g_start == 3 or not pd.isna(h23)) else 2)
            ax_traj.hlines(
                y=0.5, xmin=current_x, xmax=x_max, colors=ablett_colors[final_g], linewidth=30, capstyle="butt"
            )

        if show_segment_dividers and transition_points:
            ax_traj.vlines(x=transition_points, ymin=0.0, ymax=1.0, colors="white", linewidth=1.5, zorder=10)

        patient_alerts = alerts_merged[
            (alerts_merged["patient_id"] == pid)
            & (alerts_merged["rel_hours"] >= 0)
            & (alerts_merged["rel_hours"] <= x_max)
        ]
        alert_hours = patient_alerts["rel_hours"].values

        if len(alert_hours) > 0:
            ax_traj.vlines(x=alert_hours, ymin=0, ymax=1, colors="black", linewidth=0.5, alpha=0.8, zorder=15)

        ax_traj.set_ylim(0, 1)
        ax_traj.margins(y=0)
        ax_traj.set_yticks([])
        ax_traj.set_ylabel(short_label(pid), rotation=0, ha="right", va="center", fontsize=10)
        for spine in ["top", "right", "bottom"]:
            ax_traj.spines[spine].set_visible(False)
        ax_traj.spines["left"].set_visible(True)
        ax_traj.spines["left"].set_color("black")
        ax_traj.spines["left"].set_linewidth(1.0)

        if len(alert_hours) >= 2:
            try:
                kde = gaussian_kde(alert_hours, bw_method=0.15)
                density = kde(x_grid)
                ax_kde.fill_between(x_grid, density, color="black", alpha=0.15)
                ax_kde.plot(x_grid, density, color="black", linewidth=0.7, alpha=0.6)
            except Exception:  # noqa: S110 - KDE can fail for degenerate alert sets; skip its overlay
                pass

        ax_kde.autoscale_view()
        y_top = ax_kde.get_ylim()[1]
        if y_top == 0:
            y_top = 1
        gap_factor = 1.15
        ax_kde.set_ylim(0, y_top * gap_factor)
        ax_kde.margins(x=0, y=0)
        ax_kde.set_yticks([])
        ax_kde.set_ylabel("", rotation=0)
        for spine in ["top", "right", "bottom"]:
            ax_kde.spines[spine].set_visible(False)
        ax_kde.spines["left"].set_visible(True)
        ax_kde.spines["left"].set_color("black")
        ax_kde.spines["left"].set_linewidth(1.0)

        if i == n_patients - 1:
            ax_traj.spines["bottom"].set_visible(True)
            ax_traj.spines["bottom"].set_color("black")

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, x_max)
        ax.tick_params(axis="x", which="both", length=0)
    axes[-1].tick_params(axis="x", which="both", length=4)
    axes[-1].set_xlabel("Hours since study enrollment", fontsize=11)

    fig.subplots_adjust(hspace=0.0, bottom=0.08)
    _save_fig(save_path)


def main_figures() -> None:
    """Generate Figure 3 panels b through h (trajectories, model probability ECDFs, reason and response bars)."""
    _use_arial_if_available()

    out_dir = conf.vietnam.figures_dir / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    alerts_with_trans, endpoints = _load_alerts_with_phase()

    # Panels g and h: alert reasons by Ablett phase.
    plot_alert_reasons_counts(alerts_with_trans, out_dir / "alert_reasons_by_ablett_counts.pdf")
    logger.info("Saved %s", out_dir / "alert_reasons_by_ablett_counts.pdf")
    plot_alert_reasons_prop(alerts_with_trans, out_dir / "alert_reasons_by_ablett_prop.pdf")
    logger.info("Saved %s", out_dir / "alert_reasons_by_ablett_prop.pdf")

    # Panels e and f: clinician response by Ablett phase.
    clin = alerts_with_trans[alerts_with_trans["event_status"].notna()]
    plot_clinician_response_counts(clin, out_dir / "clinician_response_by_ablett_counts.pdf")
    logger.info("Saved %s", out_dir / "clinician_response_by_ablett_counts.pdf")
    plot_clinician_response_prop(clin, out_dir / "clinician_response_by_ablett_prop.pdf")
    logger.info("Saved %s", out_dir / "clinician_response_by_ablett_prop.pdf")

    # By hour line plots.
    plot_alert_reasons_by_hour_ablett(alerts_with_trans, out_dir / "alert_reasons_by_hour_ablett.pdf")
    logger.info("Saved %s", out_dir / "alert_reasons_by_hour_ablett.pdf")
    plot_clinician_response_by_hour_ablett(clin, out_dir / "clinician_response_by_hour_ablett.pdf")
    logger.info("Saved %s", out_dir / "clinician_response_by_hour_ablett.pdf")

    # Panels c and d: model probability ECDFs.
    day_mask = alerts_with_trans["time_of_day"].str.startswith("Day")
    night_mask = alerts_with_trans["time_of_day"].str.startswith("Night")
    weekday_mask = alerts_with_trans["day_type"] == "Weekday"
    weekend_mask = alerts_with_trans["day_type"] == "Weekend"

    plot_model_prob_cumulative(
        alerts_with_trans,
        out_dir / "model_probability_cumulative_day_night.pdf",
        day_mask,
        night_mask,
        "Day",
        "Night",
        C_ACCENT,
        C_BLUE,
    )
    logger.info("Saved %s", out_dir / "model_probability_cumulative_day_night.pdf")
    plot_model_prob_cumulative(
        alerts_with_trans,
        out_dir / "model_probability_cumulative_weekday_weekend.pdf",
        weekday_mask,
        weekend_mask,
        "Weekday",
        "Weekend",
        C_ACCENT,
        C_BLUE,
    )
    logger.info("Saved %s", out_dir / "model_probability_cumulative_weekday_weekend.pdf")

    # Panel b: patient trajectories (base and alert annotated).
    _require_cleaned(conf.vietnam.cleaned_clinical_notes_path, "clinical notes")
    clinical = pd.read_csv(conf.vietnam.cleaned_clinical_notes_path, parse_dates=["event_datetime"])
    for col in ("hour", "day_of_week"):
        if col in clinical.columns:
            clinical[col] = pd.to_numeric(clinical[col], errors="coerce")
    if "date" in clinical.columns:
        clinical["date"] = pd.to_datetime(clinical["date"], errors="coerce").dt.date
    logger.info("Loaded %d cleaned clinical events.", len(clinical))
    logger.info("Loaded %d cleaned patient endpoints.", len(endpoints))

    patient_trajectories_dir = conf.vietnam.figures_dir / "patient_trajectories"
    patient_trajectories_dir.mkdir(parents=True, exist_ok=True)

    plot_patient_trajectories(endpoints, patient_trajectories_dir / "patient_trajectories.pdf")
    logger.info("Saved %s", patient_trajectories_dir / "patient_trajectories.pdf")

    # Alert frame for the annotated trajectory (without phase columns).
    alerts = pd.read_csv(
        conf.vietnam.cleaned_alert_records_path,
        parse_dates=["alert_datetime", "assessment_datetime"],
    )
    plot_patient_trajectories_with_alerts(
        endpoints, alerts, patient_trajectories_dir / "patient_trajectories_with_alerts.pdf"
    )
    logger.info("Saved %s", patient_trajectories_dir / "patient_trajectories_with_alerts.pdf")


# ---------------------------------------------------------------------------
# 3-4. Waveform panels
# ---------------------------------------------------------------------------

# Per figure cohort colours, which differ between the two waveform figures.
WAVEFORM_PANEL_C_ORANGE = "#e07a5f"
WAVEFORM_PANEL_C_GREEN = "#81b29a"
BEAT_OVERLAY_C_ORANGE = "#F28E2B"
BEAT_OVERLAY_C_GREEN = "#59A14F"

# matplotlib style shared by both waveform figures.
_WAVEFORM_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
}

# Cache of loaded PPG dataframes (keyed by file path).
_ppg_cache: dict = {}


def _bandpass(sig, lo=0.5, hi=8.0, fs=FS, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def _build_file_index(subject: str) -> list[tuple]:
    """Index PPG CSVs for a subject as (first_dt, last_dt, path), tz Asia/Bangkok."""
    idx = []
    for fp in sorted((conf.vietnam.waveforms_dir / subject).rglob("PPG/SmartCareCsv_*.csv")):
        if fp.stat().st_size < 5000:
            continue
        try:
            df = pd.read_csv(fp, usecols=["timestamp"])
            first = pd.to_datetime(df["timestamp"].iloc[0], utc=True).tz_convert("Asia/Bangkok")
            last = pd.to_datetime(df["timestamp"].iloc[-1], utc=True).tz_convert("Asia/Bangkok")
            idx.append((first, last, fp))
        except Exception:  # noqa: S112 - skip unreadable / malformed PPG files
            continue
    return idx


def _get_df(fpath) -> pd.DataFrame:
    if fpath not in _ppg_cache:
        df = pd.read_csv(fpath, usecols=["timestamp", "pleth"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Bangkok")
        _ppg_cache[fpath] = df
    return _ppg_cache[fpath]


def _load_chunk(findex, target_dt, half_rows):
    for first, last, fp in findex:
        if first <= target_dt <= last:
            df = _get_df(fp)
            diffs = (df["timestamp"] - target_dt).abs()
            c = diffs.idxmin()
            if diffs.loc[c].total_seconds() > 120:
                return None
            return df.iloc[max(0, c - half_rows) : min(len(df), c + half_rows)]
    return None


def _stitch(chunk) -> np.ndarray:
    signal = []
    for val in chunk["pleth"]:
        try:
            signal.extend(json.loads(val) if isinstance(val, str) else [val])
        except Exception:
            signal.extend([np.nan] * 100)
    signal = np.array(signal, dtype=float)
    return signal[~np.isnan(signal)]


def _get_resampled_beats(findex, target_dt) -> list:
    """Return a list of resampled, normalised beats for a window."""
    chunk = _load_chunk(findex, target_dt, WINDOW_ROWS // 2)
    if chunk is None:
        return []
    signal = _stitch(chunk)
    if len(signal) < 500:
        return []
    try:
        sig = _bandpass(signal)
    except Exception:
        return []
    sig_norm = (sig - sig.mean()) / (sig.std() + 1e-9)
    peaks, _ = find_peaks(sig_norm, distance=int(0.33 * FS), prominence=0.4)
    if len(peaks) < 4:
        return []
    beats = []
    for i in range(len(peaks) - 1):
        beat = sig[peaks[i] : peaks[i + 1]]
        if len(beat) < 20 or len(beat) > 200:
            continue
        bmin, bmax = beat.min(), beat.max()
        if bmax <= bmin:
            continue
        beats.append(resample((beat - bmin) / (bmax - bmin), BEAT_RESAMPLE))
    return beats if len(beats) >= 3 else []


def _get_window_summary_beat(findex, target_dt, use_median=False):
    """Return the mean or median resampled beat for a window."""
    beats = _get_resampled_beats(findex, target_dt)
    if not beats:
        return None
    arr = np.array(beats)
    return np.median(arr, axis=0) if use_median else np.mean(arr, axis=0)


def _resolve_waveform_subject(subject: str | None) -> str:
    """Resolve the subject, warning if the baseline windows do not apply to it."""
    resolved = subject or conf.vietnam.waveform_subject
    if resolved != conf.vietnam.waveform_subject:
        logger.warning(
            "VIETNAM_BASELINE_PERIODS are hand-curated for subject %s; they are not "
            "meaningful for requested subject %s.",
            conf.vietnam.waveform_subject,
            resolved,
        )
    return resolved


def _load_subject_alerts(subject: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame]:
    """Read the raw alert records (sep=';') and derive the per subject alert frame.

    Returns ``(subject_alerts, ab4_dt, confirmed_alerts)``. ``ab4_dt`` is the earliest
    confirmed Ablett 4 alert time. Uses the capital ``studyID`` column and numeric
    ``current_ablett``, and intentionally does NOT route through the cleaning transforms.
    """
    alerts = pd.read_csv(conf.vietnam.alert_records_path, sep=";")
    sa = alerts[alerts["studyID"] == subject].copy()
    sa["alert_dt"] = pd.to_datetime(sa["date_alert"] + " " + sa["time_alert"]).dt.tz_localize("Asia/Bangkok")

    conf_alerts = sa[(sa["possible_reason"] == "Change Ablett score") & (sa["current_ablett"] == 4)]
    ab4_dt = conf_alerts["alert_dt"].min()
    return sa, ab4_dt, conf_alerts


def _baseline_times(sa: pd.DataFrame, min_gap_sec: int = 300) -> list:
    """Expand VIETNAM_BASELINE_PERIODS into per minute timestamps >=min_gap from any alert."""
    all_alert_times = sa["alert_dt"].values
    all_alert_ts = [
        pd.Timestamp(t).tz_convert("Asia/Bangkok")
        if pd.Timestamp(t).tzinfo
        else pd.Timestamp(t).tz_localize("Asia/Bangkok")
        for t in all_alert_times
    ]

    def is_away_from_alerts(dt):
        return all(abs((a_dt - dt).total_seconds()) >= min_gap_sec for a_dt in all_alert_ts)

    baseline_times = []
    for start, end in VIETNAM_BASELINE_PERIODS:
        for dt in pd.date_range(start, end, freq="1min", tz="Asia/Bangkok"):
            if is_away_from_alerts(dt):
                baseline_times.append(dt)
    return baseline_times


def _extract_raw_waveform(findex, target_dt, waveform_sec):
    """Return (t, signal, hr) for a ``waveform_sec`` window."""
    half = waveform_sec // 2 + 1
    chunk = _load_chunk(findex, target_dt, half)
    if chunk is None:
        return None, None, None
    signal = _stitch(chunk)
    if len(signal) < 300:
        return None, None, None
    try:
        sig = _bandpass(signal)
    except Exception:
        return None, None, None
    smin, smax = sig.min(), sig.max()
    if smax > smin:
        sig = (sig - smin) / (smax - smin)
    sig_z = (sig - sig.mean()) / (sig.std() + 1e-9)
    peaks, _ = find_peaks(sig_z, distance=int(0.33 * FS), prominence=0.4)
    hr = None
    if len(peaks) >= 3:
        ibis = np.diff(peaks) / FS
        ibis = ibis[(ibis > 0.3) & (ibis < 2.0)]
        if len(ibis) >= 2:
            hr = 60.0 / np.mean(ibis)
    n = min(len(sig), waveform_sec * FS)
    return np.arange(n) / FS, sig[:n], hr


def _find_medoid(findex, times, waveform_sec, use_median=False, metric="correlation"):
    """Compute the summary beat per window, find the medoid, return its raw waveform.

    Returns ``(t, sig, hr, n_used)``.
    """
    entries = []
    for dt in times:
        mb = _get_window_summary_beat(findex, dt, use_median=use_median)
        if mb is not None:
            entries.append((dt, mb))

    if len(entries) < 2:
        if entries:
            t, sig, hr = _extract_raw_waveform(findex, entries[0][0], waveform_sec)
            return t, sig, hr, len(entries)
        return None, None, None, 0

    beat_matrix = np.array([e[1] for e in entries])
    dists = squareform(pdist(beat_matrix, metric=metric))
    medoid_idx = np.argmin(dists.sum(axis=1))
    medoid_dt = entries[medoid_idx][0]

    t, sig, hr = _extract_raw_waveform(findex, medoid_dt, waveform_sec)
    return t, sig, hr, len(entries)


def main_waveforms(subject: str | None = None) -> None:  # noqa: C901
    """Generate Figure 3 panel a (four panel raw PPG waveforms by alert cohort)."""
    warnings.filterwarnings("ignore")
    _use_arial_if_available()
    mpl.rcParams.update(_WAVEFORM_RC)

    subject = _resolve_waveform_subject(subject)
    out_dir = conf.vietnam.figures_dir / "waveform_panel"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not conf.vietnam.alert_records_path.exists():
        msg = f"Alert records file not found: {conf.vietnam.alert_records_path}"
        raise FileNotFoundError(msg)
    subject_dir = conf.vietnam.waveforms_dir / subject
    if not subject_dir.exists():
        msg = f"Waveform directory not found for subject {subject}: {subject_dir}"
        raise FileNotFoundError(msg)

    waveform_sec = 10  # extract at max length, trim for 5s figure

    findex = _build_file_index(subject)

    sa, ab4_dt, conf_alerts = _load_subject_alerts(subject)

    pre = sa[(sa["possible_reason"].isin(["False Alert", "No Idea"])) & (sa["current_ablett"] == 3)].copy()
    pre["h"] = (ab4_dt - pre["alert_dt"]).dt.total_seconds() / 3600
    dismissed = pre[(pre["h"] > 0) & (pre["h"] <= 6)]

    # Cohort c: dismissed alerts at Ablett 3, >6h from AB4.
    exclude_reasons = {"Change Ablett score", "Noised data"}
    for r in sa["possible_reason"].dropna().unique():
        if "noised" in r.lower() or "noise" in r.lower():
            exclude_reasons.add(r)
    all_dismissed = sa[
        (~sa["possible_reason"].isin(exclude_reasons)) & (sa["current_ablett"] == 3) & (sa["possible_reason"].notna())
    ].copy()
    all_dismissed["h_abs"] = ((all_dismissed["alert_dt"] - ab4_dt).dt.total_seconds() / 3600).abs()
    true_fa = all_dismissed[all_dismissed["h_abs"] > 6]

    # Cohort d: all alert free Ablett 3 PPG periods.
    baseline_times = _baseline_times(sa)
    logger.info(
        "Baseline: %d alert-free windows across %d day-periods",
        len(baseline_times),
        len(VIETNAM_BASELINE_PERIODS),
    )

    cohorts = [
        ("Confirmed alerts", list(conf_alerts["alert_dt"]), WAVEFORM_PANEL_C_ORANGE),
        ("Dismissed alerts", list(dismissed["alert_dt"]), WAVEFORM_PANEL_C_ORANGE),
        ("False positive alert", list(true_fa["alert_dt"]), WAVEFORM_PANEL_C_GREEN),
        ("Non-alert baseline", baseline_times, WAVEFORM_PANEL_C_GREEN),
    ]

    # Compute medoid per cohort (median beat, correlation).
    medoid_panels = []
    for title, times, colour in cohorts:
        t, sig, hr, n_used = _find_medoid(findex, times, waveform_sec, use_median=True, metric="correlation")
        if hr:
            logger.info("  %-25s (%d windows) -> medoid from %d windows, HR=%.0f", title, len(times), n_used, hr)
        else:
            logger.info("  %-25s (%d windows) -> no data", title, len(times))
        medoid_panels.append((title, t, sig, hr, colour, n_used))

    # Generate 5s and 10s figures (2x2 layout).
    for ws, suffix in [(5, "5s"), (10, "10s")]:
        fig, axes = plt.subplots(2, 2, figsize=(5, 3.5))
        fig.subplots_adjust(hspace=0.40, wspace=0.25, left=0.10, right=0.97, top=0.95, bottom=0.10)

        for i, (title, t, sig, hr, colour, _n_used) in enumerate(medoid_panels):
            ax = axes[i // 2, i % 2]

            if t is not None and sig is not None:
                xlim = min(ws, t[-1])
                ax.plot(t[: int(xlim * FS)], sig[: int(xlim * FS)], color=colour, lw=1.5, alpha=0.9)
                ax.set_xlim(0, ws)
                ax.set_ylim(-0.05, 1.05)
                ax.set_yticks([0, 0.5, 1.0])
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#999")

            if i % 2 == 0:
                ax.set_ylabel("Normalized PPG waveform")
            if i // 2 == 1:
                ax.set_xlabel("Time (s)")

            hr_str = f" (HR ~ {hr:.0f} bpm)" if hr else ""
            ax.set_title(title + hr_str, fontsize=6, fontweight="bold", color="black", loc="left", pad=3)

        _save_fig(out_dir / f"fig_disagreement_metrics_{suffix}.pdf")
        logger.info("  Saved %s", out_dir / f"fig_disagreement_metrics_{suffix}")


def _get_window_median_beat(findex, target_dt):
    """Median resampled beat for a window (beat_overlay convention)."""
    return _get_window_summary_beat(findex, target_dt, use_median=True)


def _extract_medoid_beat(findex, times):  # noqa: C901
    """Find the medoid window, then extract its middle beat in real time (ms).

    Returns ``(t_ms, beat_normalised, hr, n_windows)``.
    """
    entries = []
    for dt in times:
        mb = _get_window_median_beat(findex, dt)
        if mb is not None:
            entries.append((dt, mb))
    if len(entries) < 2:
        if not entries:
            return None, None, None, 0
        dt = entries[0][0]
    else:
        beat_matrix = np.array([e[1] for e in entries])
        dists = squareform(pdist(beat_matrix, metric="correlation"))
        medoid_idx = np.argmin(dists.sum(axis=1))
        dt = entries[medoid_idx][0]

    chunk = _load_chunk(findex, dt, WINDOW_ROWS // 2)
    if chunk is None:
        return None, None, None, len(entries)
    signal = _stitch(chunk)
    if len(signal) < 300:
        return None, None, None, len(entries)
    try:
        sig = _bandpass(signal)
    except Exception:
        return None, None, None, len(entries)

    sig_z = (sig - sig.mean()) / (sig.std() + 1e-9)
    peaks, _ = find_peaks(sig_z, distance=int(0.33 * FS), prominence=0.4)
    if len(peaks) < 4:
        return None, None, None, len(entries)

    mid = len(peaks) // 2
    beat = sig[peaks[mid] : peaks[mid + 1]]
    if len(beat) < 20:
        return None, None, None, len(entries)

    bmin, bmax = beat.min(), beat.max()
    if bmax > bmin:
        beat = (beat - bmin) / (bmax - bmin)

    t_ms = np.arange(len(beat)) / FS * 1000
    hr = 60.0 / (len(beat) / FS)
    return t_ms, beat, hr, len(entries)


def main_beat_overlay(subject: str | None = None) -> None:
    """Generate Figure 3 panel i (median beat overlay: dismissed alert vs. baseline)."""
    warnings.filterwarnings("ignore")
    _use_arial_if_available()
    rc = dict(_WAVEFORM_RC)
    rc["font.sans-serif"] = ["Arial"]
    mpl.rcParams.update(rc)

    subject = _resolve_waveform_subject(subject)
    out_dir = conf.vietnam.figures_dir / "beat_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not conf.vietnam.alert_records_path.exists():
        msg = f"Alert records file not found: {conf.vietnam.alert_records_path}"
        raise FileNotFoundError(msg)
    subject_dir = conf.vietnam.waveforms_dir / subject
    if not subject_dir.exists():
        msg = f"Waveform directory not found for subject {subject}: {subject_dir}"
        raise FileNotFoundError(msg)

    findex = _build_file_index(subject)

    sa, ab4_dt, _conf_alerts = _load_subject_alerts(subject)

    # Cohort b: dismissed <=6h.
    pre = sa[(sa["possible_reason"].isin(["False Alert", "No Idea"])) & (sa["current_ablett"] == 3)].copy()
    pre["h"] = (ab4_dt - pre["alert_dt"]).dt.total_seconds() / 3600
    dismissed = pre[(pre["h"] > 0) & (pre["h"] <= 6)]

    # Cohort d: baseline.
    baseline_times = _baseline_times(sa)

    t_b, beat_b, hr_b, n_b = _extract_medoid_beat(findex, list(dismissed["alert_dt"]))
    t_d, beat_d, hr_d, n_d = _extract_medoid_beat(findex, baseline_times)

    if hr_b is not None:
        logger.info("Dismissed: HR=%.0f, medoid of %d windows", hr_b, n_b)
    else:
        logger.info("Dismissed: no data (medoid of %d windows)", n_b)
    if hr_d is not None:
        logger.info("Baseline:  HR=%.0f, medoid of %d windows", hr_d, n_d)
    else:
        logger.info("Baseline:  no data (medoid of %d windows)", n_d)

    if t_b is None or t_d is None:
        msg = "Could not extract median beats for the beat-overlay figure (insufficient PPG windows)."
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=(1.5, 2.8))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.25)

    ax.plot(t_b, beat_b, color=BEAT_OVERLAY_C_ORANGE, lw=2.0, label="Dismissed alert")
    ax.plot(t_d, beat_d, color=BEAT_OVERLAY_C_GREEN, lw=2.0, label="Non-alert baseline")

    x_max = max(t_b.max(), t_d.max())
    ax.set_xlim(0, x_max)

    ax.set_xlabel("Time from systolic peak (ms)")
    ax.set_ylabel("Normalized PPG amplitude")
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#ccc")
    ax.set_axisbelow(True)

    ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

    _save_fig(out_dir / "beat_overlay.pdf")
    logger.info("Saved to %s", out_dir)


# ---------------------------------------------------------------------------
# 5. Patient statistics
# ---------------------------------------------------------------------------


def _fmt_td(td) -> str:
    """Format a Timedelta as e.g. '3d 4h 12m' (NA-safe)."""
    if pd.isna(td):
        return "NA"
    sec = float(td.total_seconds())
    sign = ""
    if sec < 0:
        sign = "-"
        sec = -sec
    days, rem = divmod(sec, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{int(days)}d")
    if hours or not parts:
        parts.append(f"{int(hours)}h")
    parts.append(f"{int(minutes)}m")
    return sign + " ".join(parts)


def _print_patient_timeline(row: pd.Series) -> None:
    pid = row["patient_id"]
    admission = row["enrollment_datetime"]
    earliest = row["earliest_alert"]
    latest = row["latest_alert"]

    labeled = [
        ("admission (enrollment from endpoints)", admission),
        ("earliest alert", earliest),
        ("latest alert", latest),
    ]
    ordered = sorted([(lab, t) for lab, t in labeled if pd.notna(t)], key=lambda x: x[1])

    logger.info("--- %s ---", pid)
    logger.info("Chronological order:")
    for lab, t in ordered:
        logger.info("  %s: %s", lab, t)

    if pd.notna(admission) and pd.notna(latest):
        delta_ad_latest = latest - admission
        logger.info("  admission -> latest alert: %s  (%s)", _fmt_td(delta_ad_latest), delta_ad_latest)
    else:
        logger.info("  admission -> latest alert: NA (missing admission or no alerts)")

    if pd.notna(earliest) and pd.notna(latest):
        delta_el = latest - earliest
        logger.info("  earliest -> latest alert: %s  (%s)", _fmt_td(delta_el), delta_el)
    else:
        logger.info("  earliest -> latest alert: NA (no alerts)")


def main_stats() -> None:
    """Print Vietnam cohort statistics for Table 1 (n patients, follow-up durations)."""
    endpoints_path = conf.vietnam.cleaned_endpoints_path
    alerts_path = conf.vietnam.cleaned_alert_records_path

    _require_cleaned(endpoints_path, "endpoints")
    _require_cleaned(alerts_path, "alert records")

    logger.info("Endpoints: %s", endpoints_path)
    logger.info("Alerts:    %s", alerts_path)

    endpoints = pd.read_csv(
        endpoints_path,
        parse_dates=[
            "enrollment_datetime",
            "ablett_2_3_transition_datetime",
            "ablett_3_4_transition_datetime",
            "significant_event_1_datetime",
            "significant_event_2_datetime",
        ],
    )

    alerts = pd.read_csv(alerts_path, parse_dates=["alert_datetime", "assessment_datetime"])
    alerts = alerts[alerts["alert_datetime"].notna()].copy()

    logger.info("Patients (endpoints): %d", endpoints["patient_id"].nunique())
    logger.info("Alert rows (valid alert_datetime): %d", len(alerts))

    # Age and sex summary for Table 1, sourced from the cleaned clinical notes
    # (patient_yob and patient_gender). Age is computed relative to each patient's
    # enrollment year from the endpoints sheet (no arbitrary reference year).
    if conf.vietnam.cleaned_clinical_notes_path.exists():
        clinical = pd.read_csv(conf.vietnam.cleaned_clinical_notes_path)
        if {"patient_yob", "patient_gender"}.issubset(clinical.columns):
            demo = clinical.dropna(subset=["patient_yob"]).drop_duplicates("patient_id")[
                ["patient_id", "patient_yob", "patient_gender"]
            ]
            enroll_year = endpoints.assign(enroll_year=endpoints["enrollment_datetime"].dt.year)[
                ["patient_id", "enroll_year"]
            ]
            demo = demo.merge(enroll_year, on="patient_id", how="left")
            demo["age"] = demo["enroll_year"] - demo["patient_yob"]
            ages = demo["age"].dropna()
            gender_counts = demo["patient_gender"].str.lower().value_counts().to_dict()
            if not ages.empty:
                logger.info(
                    "Demographics (n=%d): age at enrollment median %.0f [%.0f-%.0f]; sex %s",
                    demo["patient_id"].nunique(),
                    float(ages.median()),
                    float(ages.min()),
                    float(ages.max()),
                    gender_counts,
                )

    alert_bounds = alerts.groupby("patient_id", as_index=False).agg(
        earliest_alert=("alert_datetime", "min"), latest_alert=("alert_datetime", "max")
    )
    trajectory = endpoints[["patient_id", "enrollment_datetime"]].merge(alert_bounds, on="patient_id", how="left")

    for _, r in trajectory.sort_values("patient_id").iterrows():
        _print_patient_timeline(r)

    trajectory["admission_to_latest"] = trajectory["latest_alert"] - trajectory["enrollment_datetime"]
    trajectory["earliest_to_latest"] = trajectory["latest_alert"] - trajectory["earliest_alert"]

    valid_follow = trajectory["admission_to_latest"].dropna()

    logger.info("=== Cohort: admission -> latest alert ===")
    logger.info("Patients with at least one alert and known enrollment: %d", len(valid_follow))
    if len(valid_follow):
        days = valid_follow.dt.total_seconds() / 86400
        logger.info("Min follow-up (admission -> latest):  %s  (%s)", _fmt_td(valid_follow.min()), valid_follow.min())
        logger.info("Max follow-up (admission -> latest):  %s  (%s)", _fmt_td(valid_follow.max()), valid_follow.max())
        logger.info("Distribution in days:\n%s", days.describe().to_string())
    else:
        logger.info("No complete rows for admission -> latest.")
