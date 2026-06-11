"""Switzerland pilot analysis.

Figure generation code adapted from the ETH Zurich Ratschlab ai4icu project
(github.com/ratschlab/projects2026-ai4icu-medlog).

This module reproduces the lettered panels of paper Figure 2 plus the Table 1
"Bern" cohort row:

  * ``main_early_alarms``    -> panels b, c (alarm burden by failure group) and
                               d (hourly alarm rate over the first 72 h).
  * ``main_feature_recency`` -> panels e, f (model prediction vs. arterial
                               lactate recency: bar plot and joint heatmap).
  * ``main_human_response``  -> panels g, h (lab order density and cumulative
                               incidence after circulatory alarms; lifelines
                               Cox PH and log-rank).
  * ``main_fairness``        -> panels i, j, k (sex/age AUROC disparity, Page's
                               CUSUM, and per task subgroup AUROC bars).
  * ``main_stats``           -> console cohort statistics (Table 1, Bern row).

All data is read from ``conf.beacon.parquet_path`` (a local assembled parquet;
see the README). The four commands other than ``fairness`` require that file and
raise ``FileNotFoundError`` when it is absent. ``main_fairness`` falls back to a
synthetic assembled dataframe so the command runs out of the box.
"""

import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import conf

logger = logging.getLogger(__name__)


def _setup_fonts() -> None:
    """Use Arial if it is available, otherwise fall back to the default font."""
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    if "Arial" in available:
        matplotlib.rcParams["font.family"] = "Arial"
    else:
        logger.info("Arial font not available; using matplotlib default font.")


# Shared color palette used across the BEACON panels.
COLOR_PALETTE = {
    "red": {"dark": "#E64B35", "light": "#F39B7F"},
    "blue": {"light": "#4DBBD5", "dark": "#8491B4"},
    "green": {"dark": "#00A087", "light": "#91D1C2"},
    "brown": {"dark": "#7E6148", "light": "#B09C85"},
    "purple": {"dark": "#6A3D9A", "light": "#CAB2D6"},
    "yellow": {"dark": "#D4B013", "light": "#F3E08C"},
    "accent_blue": "#3C5488",
    "accent_red": "#DC0000",
}


def _save_fig(fig, path: Path, dpi: int = 600) -> None:
    """Save a figure as PDF, SVG, and PNG (600 dpi) alongside each other."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path.with_suffix(".pdf")
    svg_path = path.with_suffix(".svg")
    png_path = path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi, format="png")
    logger.info(f"Saved figure to {pdf_path}, {svg_path}, {png_path}")


def _load_assembled() -> pd.DataFrame:
    """Load the assembled parquet, raising a clear error if it is missing.

    Returns the dataframe with a tz-aware ``datetime`` column.
    """
    path = conf.beacon.parquet_path
    if not path.exists():
        msg = (
            f"assembled parquet not found at {path}; see README for how to obtain "
            "the local BEACON data file (it is not included in this repository)."
        )
        raise FileNotFoundError(msg)
    logger.info(f"Loading assembled data from {path}")
    assembled = pd.read_parquet(path)
    assembled["datetime"] = pd.to_datetime(assembled["datetime"], utc=True)
    return assembled


#  Figure 2 b, c, d: early alarms

# Event columns used to stratify patients. Each alarm type is paired with its
# clinically matched event label so the stratification captures patients who
# experienced the corresponding failure at any point.
CIRC_ALARM_COL = "circ_alarm"
RESP_ALARM_COL = "resp_alarm"
CIRC_EVENT_COL = "Dynamic_CircFailure_Event"
RESP_EVENT_COL = "Dynamic_RespFailure_mid_Event"


def bootstrap_alarm_counts(
    df: pd.DataFrame,
    alarm_col: str,
    event_group_col: str,
    group_val: bool,
    all_bins: list,
    n_bootstrap: int = 500,
    alpha: float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Bootstrap patient level alarm counts per time bin.

    Each iteration samples ``n_patients`` patients with replacement, then sums
    their alarm counts per bin. This preserves the correlation structure across
    bins (a patient draw affects all bins simultaneously), so any downstream
    aggregation over bins can still use the joint bootstrap distribution.

    Returns ``(mean, lo, hi, n_patients, boot_totals)`` where ``boot_totals`` has
    shape ``(n_bootstrap, len(all_bins))``.
    """
    group_df = df[df[event_group_col] == group_val].copy()
    patients = group_df["patient_id"].unique()
    n_patients = len(patients)

    # Build alarm sums per patient per bin (patients absent in a bin contribute 0).
    pat_bin_counts = (
        group_df.assign(**{alarm_col: group_df[alarm_col].astype(int)})
        .groupby(["patient_id", "bin"])[alarm_col]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=all_bins, fill_value=0)
    )
    pat_bin_counts = pat_bin_counts.reindex(patients, fill_value=0)
    counts_matrix = pat_bin_counts.values

    rng = np.random.default_rng(seed)
    boot_totals = np.zeros((n_bootstrap, len(all_bins)))
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_patients, size=n_patients)
        boot_totals[i] = counts_matrix[idx].sum(axis=0)

    mean = boot_totals.mean(axis=0)
    lo = np.percentile(boot_totals, (1 - alpha) / 2 * 100, axis=0)
    hi = np.percentile(boot_totals, (1 + alpha) / 2 * 100, axis=0)

    return mean, lo, hi, n_patients, boot_totals


def main_early_alarms() -> Path:
    """Generate Figure 2 panels b, c (``early_alarms_by_group``) and d
    (``early_alarms_by_admission_time``).

    Returns the figures directory.
    """
    _setup_fonts()
    assembled = _load_assembled()
    assembled["admission_time"] = pd.to_datetime(assembled["admission_time"], utc=True)

    figures_dir = conf.beacon.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Analysis parameters
    cutoff_days = 3
    bin_hours = 1
    n_bootstrap = 500
    ci_alpha = 0.95
    tick_every_hours = 12
    early_window_hours = 1
    cutoff_hours = cutoff_days * 24

    # Prepare data
    assembled["hours_since_admission"] = assembled["rel_datetime"].astype(float) / 3600.0
    df = assembled[assembled["hours_since_admission"].between(0, cutoff_hours)].copy()
    df["bin"] = (df["hours_since_admission"] // bin_hours).astype(int)
    all_bins = list(range(cutoff_hours // bin_hours))

    # Event flags per patient, based on the full stay, not just the filtered window.
    circ_event_pids = set(assembled.loc[assembled[CIRC_EVENT_COL].astype(float) == 1.0, "patient_id"].unique())
    resp_event_pids = set(assembled.loc[assembled[RESP_EVENT_COL].astype(float) == 1.0, "patient_id"].unique())
    df["circ_event_group"] = df["patient_id"].isin(circ_event_pids)
    df["resp_event_group"] = df["patient_id"].isin(resp_event_pids)
    df[CIRC_ALARM_COL] = df[CIRC_ALARM_COL].fillna(False).astype(bool)
    df[RESP_ALARM_COL] = df[RESP_ALARM_COL].fillna(False).astype(bool)

    n_total = df["patient_id"].nunique()
    logger.info(f"Patients in analysis window: {n_total}")

    # Compute bootstrap results
    results = {}
    for alarm_col, event_col, label in [
        (CIRC_ALARM_COL, "circ_event_group", "circ"),
        (RESP_ALARM_COL, "resp_event_group", "resp"),
    ]:
        for group_val, group_label in [(True, "event"), (False, "no_event")]:
            mean, lo, hi, n_pat, boot_totals = bootstrap_alarm_counts(
                df,
                alarm_col=alarm_col,
                event_group_col=event_col,
                group_val=group_val,
                all_bins=all_bins,
                n_bootstrap=n_bootstrap,
                alpha=ci_alpha,
            )
            key = f"{label}_{group_label}"
            results[key] = {"mean": mean, "lo": lo, "hi": hi, "n_patients": n_pat, "boot_totals": boot_totals}
            logger.info(f"  [{key}]  n={n_pat}  peak mean={mean.max():.1f}")

    # Bin indices that fall entirely within the early window.
    early_bin_indices = [b for b in all_bins if (b + 1) * bin_hours <= early_window_hours]
    lo_pct = (1 - ci_alpha) / 2 * 100
    hi_pct = (1 + ci_alpha) / 2 * 100

    # Panels b + c: alarm burden by failure group (combined figure)
    _plot_early_alarms_by_group(results, early_bin_indices, lo_pct, hi_pct, figures_dir)

    # Panel d: hourly alarm rate over the first 72 h
    _plot_early_alarms_by_admission_time(
        results,
        all_bins,
        bin_hours,
        cutoff_hours,
        cutoff_days,
        tick_every_hours,
        ci_alpha,
        n_bootstrap,
        figures_dir,
    )

    return figures_dir


def _plot_early_alarms_by_group(results, early_bin_indices, lo_pct, hi_pct, figures_dir) -> None:
    """Panels b and c: mean alarms per patient (b) and fraction in first hour (c).

    Rendered as a single figure with two grouped bar panels, one per quantity,
    stratified by organ failure status.
    """
    group_labels = ["Patients without organ failure", "Patients with organ failure"]
    colors = {
        "Patients without organ failure": "#507DBC",
        "Patients with organ failure": "#011936",
    }
    alarm_specs = [(CIRC_ALARM_COL, "Circulatory"), (RESP_ALARM_COL, "Respiratory")]
    alarm_labels = ["Circulatory alarm", "Respiratory alarm"]

    # Panel b data: total alarms per patient (full ≤3 d window).
    per_patient = []
    # Panel c data: percentage of alarms occurring in the first hour.
    early_frac = []
    for alarm_col, alarm_label in alarm_specs:
        for group_val, group_label in [(False, group_labels[0]), (True, group_labels[1])]:
            key = f"{'circ' if 'circ' in alarm_col else 'resp'}_{'event' if group_val else 'no_event'}"
            r = results[key]
            bt = r["boot_totals"]
            boot_early = bt[:, early_bin_indices].sum(axis=1)
            boot_total = bt.sum(axis=1)
            boot_pct = np.where(boot_total > 0, boot_early / boot_total * 100, np.nan)
            boot_per_patient = boot_total / r["n_patients"]

            per_patient.append({
                "alarm": alarm_label,
                "group": group_label,
                "mean": np.nanmean(boot_per_patient),
                "lo": np.nanpercentile(boot_per_patient, lo_pct),
                "hi": np.nanpercentile(boot_per_patient, hi_pct),
            })
            early_frac.append({
                "alarm": alarm_label,
                "group": group_label,
                "mean": np.nanmean(boot_pct),
                "lo": np.nanpercentile(boot_pct, lo_pct),
                "hi": np.nanpercentile(boot_pct, hi_pct),
            })

    x = np.arange(len(alarm_labels))
    width = 0.35
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(6, 2.5))

    for ax, data, ylabel in [
        (ax_b, per_patient, "Alarms per patient\n(MedLog fields #4, #7)"),
        (ax_c, early_frac, "Alarms in first hour (%)"),
    ]:
        for i, group_label in enumerate(group_labels):
            group_data = [d for d in data if d["group"] == group_label]
            means = [d["mean"] for d in group_data]
            yerr = [
                [d["mean"] - d["lo"] for d in group_data],
                [d["hi"] - d["mean"] for d in group_data],
            ]
            offset = (i - 0.5) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=yerr,
                capsize=4,
                label=group_label,
                color=colors[group_label],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(alarm_labels, fontsize=10)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    _save_fig(fig, figures_dir / "early_alarms_by_group")
    plt.close(fig)

    for alarm_col, alarm_label in alarm_specs:
        for group_val, group_key in [(False, "no_event"), (True, "event")]:
            key = f"{'circ' if 'circ' in alarm_col else 'resp'}_{group_key}"
            n = results[key]["n_patients"]
            logger.info(f"{alarm_label} - {'Failure' if group_val else 'No failure'}: n={n}")


def _plot_early_alarms_by_admission_time(
    results,
    all_bins,
    bin_hours,
    cutoff_hours,
    cutoff_days,
    tick_every_hours,
    ci_alpha,
    n_bootstrap,
    figures_dir,
) -> None:
    """Panel d: total alarm count per bin over the first 72 h, by event status."""
    x = np.array(all_bins) * bin_hours  # left edge of each bin in hours
    group_colors = {
        "event": COLOR_PALETTE["red"]["dark"],
        "no_event": COLOR_PALETTE["blue"]["light"],
    }
    group_labels = {"event": "Event", "no_event": "No event"}
    subplot_cfg = [
        {"alarm": "circ", "title": "Circulatory alarms", "xlabel": f"Hours since admission (bin = {bin_hours} h)"},
        {"alarm": "resp", "title": "Respiratory alarms", "xlabel": f"Hours since admission (bin = {bin_hours} h)"},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=False)
    fig.suptitle(
        f"Alarm count by hours since admission\n"
        f"(stratified by event status · {ci_alpha:.0%} bootstrap CI · n_boot={n_bootstrap})",
        fontsize=13,
        y=1.02,
    )

    for ax, cfg in zip(axes, subplot_cfg):
        alarm = cfg["alarm"]
        for group_val_str in ["event", "no_event"]:
            key = f"{alarm}_{group_val_str}"
            r = results[key]
            n = r["n_patients"]
            color = group_colors[group_val_str]
            lbl = f"{group_labels[group_val_str]} (n={n})"
            ax.plot(x, r["mean"], color=color, linewidth=2, label=lbl)
            ax.fill_between(x, r["lo"], r["hi"], color=color, alpha=0.20)

        for d in range(1, cutoff_days + 1):
            ax.axvline(d * 24, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

        ax.set_title(cfg["title"], fontsize=12)
        ax.set_xlabel(cfg["xlabel"], fontsize=11)
        ax.set_ylabel("Total alarm count per bin", fontsize=11)
        ax.set_xlim(0, cutoff_hours)
        tick_positions = range(0, cutoff_hours + 1, tick_every_hours)
        tick_labels = [f"d{h // 24}" if h % 24 == 0 else f"+{h % 24}h" for h in tick_positions]
        ax.set_xticks(list(tick_positions))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.legend(fontsize=10, framealpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 50)

    fig.tight_layout()
    _save_fig(fig, figures_dir / "early_alarms_by_admission_time")
    plt.close(fig)


#  Figure 2 e, f: feature recency


def _make_recency_cmap():
    """White centred diverging colormap (deep blue to white to deep red)."""
    import matplotlib.colors as mcolors

    return mcolors.LinearSegmentedColormap.from_list(
        "white_to_blue",
        ["#203049", "#94B2E5", "#FFFFFF", "#E59FA6", "#971D2B"],
    )


def plot_recency_value_heatmap(
    df,
    target_variables,
    score_column=None,
    y_columns=None,
    stratify_timesteps_by=None,
    patient_col="patient_id",
    min_recency=0,
    max_recency=24,
    min_y=None,
    max_y=None,
    recency_unit="hours",
    n_recency_bins=24,
    n_y_bins=15,
    min_cell_count=5,
    figsize=None,
    cmap="RdYlGn_r",
    vmin=None,
    vmax=None,
    horizontal_lines=None,
    strata_label_map=None,
    x_label_map=None,
    y_label_map=None,
    colorbar_label=None,
    share_colorbar=True,
):
    """Heatmap of mean model score over (feature recency, feature value) bins.

    One column per stratum (failure vs. no failure within 8 h), one row per
    target variable. Cells are coloured by the mean ``score_column`` value.
    """
    strata_label_map = strata_label_map or {}
    x_label_map = x_label_map or {}
    y_label_map = y_label_map or {}

    if score_column is None:
        score_column = next(c for c in df.columns if c.endswith("_raw_prediction"))

    strata = (
        {
            f"{stratify_timesteps_by} = 1": df[stratify_timesteps_by] == 1,
            f"{stratify_timesteps_by} = 0": df[stratify_timesteps_by] == 0,
        }
        if stratify_timesteps_by
        else {"All": pd.Series(True, index=df.index)}
    )

    fig, axes = plt.subplots(
        len(target_variables),
        len(strata),
        figsize=figsize or (14, 6),
        squeeze=False,
        constrained_layout=True,
    )

    if vmin is None or vmax is None:
        sv = df[score_column].dropna()
        vmin = vmin if vmin is not None else sv.quantile(0.05)
        vmax = vmax if vmax is not None else sv.quantile(0.95)

    im = None
    for v_idx, var in enumerate(target_variables):
        recency_col = f"{var}_recency"
        value_col = y_columns[v_idx] if y_columns else f"{var}_value"

        curr_min_y = min_y if min_y is not None else df[value_col].min()
        curr_max_y = max_y if max_y is not None else df[value_col].quantile(0.99)

        for s_idx, (label, mask) in enumerate(strata.items()):
            ax = axes[v_idx][s_idx]
            sub = df[mask].dropna(subset=[recency_col, value_col, score_column])
            sub = sub[(sub[recency_col] >= min_recency) & (sub[recency_col] <= max_recency)]
            sub = sub[(sub[value_col] >= curr_min_y) & (sub[value_col] <= curr_max_y)]

            if len(sub) < 10:
                ax.set_visible(False)
                continue

            r_edges = np.linspace(min_recency, max_recency, n_recency_bins + 1)
            v_edges = np.linspace(curr_min_y, curr_max_y, n_y_bins + 1)
            sub = sub.assign(
                r_bin=pd.cut(sub[recency_col], bins=r_edges),
                v_bin=pd.cut(sub[value_col], bins=v_edges),
            )

            agg = sub.groupby(["v_bin", "r_bin"], observed=True)[score_column].agg(["mean", "count"])
            pivot = agg["mean"].unstack("r_bin").sort_index(ascending=False)

            im = ax.imshow(
                pivot.values,
                aspect="auto",
                origin="upper",
                extent=[min_recency, max_recency, curr_min_y, curr_max_y],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xlabel(x_label_map.get(f"{var} recency", f"{var} recency ({recency_unit})"), fontsize=14)
            ax.set_ylabel(y_label_map.get(f"{var} value", f"{var} value"), fontsize=14)
            ax.set_xticks(np.arange(min_recency, max_recency + 1, 3))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            n_pts = sub[patient_col].nunique()
            n_alerts = len(sub)
            title_template = strata_label_map.get(label, label)
            ax.set_title(title_template.format(patients=n_pts, alerts=n_alerts), fontsize=14, pad=12)

            if horizontal_lines:
                for line in horizontal_lines:
                    ax.axhline(line, color="black", lw=1.5, ls="--")
            ax.tick_params(labelsize=12)

    if share_colorbar and im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, fraction=0.03)
        cbar.set_label(colorbar_label or score_column, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    return fig


def main_feature_recency() -> Path:
    """Generate Figure 2 panel e (``recency_barplot_lactate_circ``) and panel f
    (``recency_heatmap_lactate_circ``).

    Both panels relate the circulatory model score to arterial lactate recency.
    Returns the figures directory.
    """
    _setup_fonts()
    assembled = _load_assembled()

    # Rename ``datetime`` to ``time``. Recency and value columns (e.g.
    # ``Lactate_recency``, ``Lactate_value``) are precomputed in the parquet.
    if "time" in assembled.columns and "datetime" in assembled.columns:
        assembled = assembled.drop(columns=["time"])
    scores_with_recency = assembled.rename(columns={"datetime": "time"})

    figures_dir = conf.beacon.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    custom_cmap = _make_recency_cmap()

    # Panel f: recency vs value heatmap (circulatory <- arterial lactate)
    fig_heat = plot_recency_value_heatmap(
        scores_with_recency,
        target_variables=["Lactate"],
        score_column="circ_raw_prediction",
        stratify_timesteps_by="Dynamic_CircFailure_8Hours",
        figsize=(10, 5),
        min_recency=0,
        max_recency=24,
        n_recency_bins=24,
        n_y_bins=20,
        horizontal_lines=[2.0],
        strata_label_map={
            "Dynamic_CircFailure_8Hours = 0": "No ciculatory failure within 8 hours\n({patients:,} patients, {alerts:,} MedLog records)",
            "Dynamic_CircFailure_8Hours = 1": "Circulatory failure within 8 hours\n({patients:,} patients, {alerts:,} MedLog records)",
        },
        colorbar_label="Output prediction",
        cmap=custom_cmap,
    )
    _save_fig(fig_heat, figures_dir / "recency_heatmap_lactate_circ")
    plt.close(fig_heat)

    # Panel e: mean prediction by lactate recency bin (failure timesteps)
    score_col = "circ_raw_prediction"
    recency_col = "Lactate_recency"
    max_recency = 24
    n_bins = 10

    sub = scores_with_recency[scores_with_recency["Dynamic_CircFailure_8Hours"] == 1].copy()
    sub = sub.dropna(subset=[recency_col, score_col])
    sub = sub[sub[recency_col] <= max_recency]

    bin_edges = np.linspace(0, max_recency, n_bins + 1)
    sub["bin"] = pd.cut(sub[recency_col], bins=bin_edges)
    agg_data = sub.groupby("bin", observed=True)[score_col].mean().reset_index()
    agg_data["bin_label"] = agg_data["bin"].apply(lambda x: f"{int(x.left)}")

    norm = plt.Normalize(agg_data[score_col].min(), agg_data[score_col].max())
    colors = [custom_cmap(norm(val)) for val in agg_data[score_col]]

    fig_bar, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.bar(agg_data["bin_label"], agg_data[score_col], color=colors, edgecolor="black", linewidth=0.8, zorder=4)
    ax.set_xlabel("Lactate recency (hours)", fontsize=14)
    ax.set_ylabel("Mean output prediction", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.7, zorder=0, axis="y")

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = fig_bar.colorbar(sm, ax=ax, pad=0.02, fraction=0.5)
    cbar.set_label("Output prediction", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    _save_fig(fig_bar, figures_dir / "recency_barplot_lactate_circ")
    plt.close(fig_bar)

    return figures_dir


#  Figure 2 g, h: human response, time to lab after alarm.
#  Uses lifelines (CoxPHFitter, logrank_test).

# Color palette for the styled cumulative incidence and density panels.
_PAL_HR = {
    "alarm": "#971D2B",
    "baseline": "#888888",
    "alarm_ci": "#C4727D",
    "base_ci": "#BBBBBB",
    "vline": "#000000",
}


def kaplan_meier(times, events):
    """Plain Kaplan-Meier survival estimate. Returns (km_t, km_s) arrays."""
    order = np.argsort(times)
    t = np.asarray(times)[order].astype(float)
    e = np.asarray(events)[order].astype(float)
    uniq_t = np.unique(t)
    n = len(t)
    km_t, km_s = [0.0], [1.0]
    n_at_risk = n
    for ut in uniq_t:
        mask = t == ut
        d_i = e[mask].sum()
        n_i = mask.sum()
        if n_at_risk > 0 and d_i > 0:
            km_s.append(km_s[-1] * (1.0 - d_i / n_at_risk))
            km_t.append(ut)
        elif n_at_risk > 0:
            km_s.append(km_s[-1])
            km_t.append(ut)
        n_at_risk -= n_i
    return np.array(km_t), np.array(km_s)


def analyze_alarm_km(  # noqa: C901
    scores_df,
    alarm_lab_pairs,
    max_forward_hours=12,
    max_backward_hours=12,
    min_hours_since_last_alarm=4,
    ignore_first_hours=None,
    censor_at_next_alarm=True,
    baseline_buffer_minutes=15,
    baseline_window_hours=4,
    n_bootstrap=500,
    bootstrap_seed=42,
    past_alarm_window_hours=12,
    cox_include_n_past_alarms=True,
    patient_col="patientid",
):
    """Bidirectional analysis of lab test timing around alarm onsets.

    For each alarm onset (rising edge), compute the time until the next lab
    observation, censoring at the analysis window and at competing alarms. A
    matched baseline is drawn from each patient's non alarm timesteps. Computes a
    forward log-rank test and a clustered Cox PH model (hazard ~ is_alarm +
    time_since_last_lab [+ n_past_alarms]).

    Returns a dict keyed by alarm label with the dataframe per onset, the fitted
    Cox model, the log-rank result, and forward KM curves.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test

    rng = np.random.RandomState(bootstrap_seed)
    baseline_buffer_h = baseline_buffer_minutes / 60.0

    scores = scores_df.copy()
    scores["datetime"] = pd.to_datetime(scores["datetime"])
    if scores["datetime"].dt.tz is not None:
        scores["datetime"] = scores["datetime"].dt.tz_convert(None)
    if ignore_first_hours is not None and "rel_datetime" in scores.columns:
        scores = scores[scores["rel_datetime"] >= ignore_first_hours * 3600].copy()

    results_all = {}

    for pair in alarm_lab_pairs:
        alarm_col = pair["alarm_col"]
        label = pair.get("label", pair["lab_col"])
        ttn_col = pair["ttn_col"]
        ttl_col = pair["ttl_col"]
        alarm_label = alarm_col.replace("_alarm", "").replace("_", " ").upper()

        skip = False
        for c in [alarm_col, ttn_col, ttl_col]:
            if c not in scores.columns:
                logger.warning(f"[SKIP] {c} not in data")
                skip = True
        if skip:
            continue

        # Detect alarm onsets
        needed = [patient_col, "datetime", alarm_col, ttn_col, ttl_col]
        s = scores[needed].sort_values([patient_col, "datetime"])
        s["prev_alarm"] = s.groupby(patient_col)[alarm_col].shift(1).fillna(0)
        onset_mask = (s[alarm_col] == 1) & (s["prev_alarm"] == 0)

        all_onsets = s.loc[onset_mask].copy()
        all_onset_times_by_pid = {pid: np.sort(grp["datetime"].values) for pid, grp in all_onsets.groupby(patient_col)}

        def _count_past_alarms(dt, pid, window_hours=past_alarm_window_hours, _times=all_onset_times_by_pid):
            if pid not in _times:
                return 0
            atimes = _times[pid]
            window = np.timedelta64(int(window_hours * 3600), "s")
            return int(((atimes < dt) & (atimes >= dt - window)).sum())

        # Filter onsets by min gap
        onsets = all_onsets.copy()
        if min_hours_since_last_alarm is not None:
            onsets = onsets.sort_values([patient_col, "datetime"])
            onsets["prev_onset_t"] = onsets.groupby(patient_col)["datetime"].shift(1)
            gap_h = (onsets["datetime"] - onsets["prev_onset_t"]).dt.total_seconds() / 3600
            onsets = onsets[gap_h.isna() | (gap_h >= min_hours_since_last_alarm)]
            onsets = onsets.drop(columns=["prev_onset_t"])

        if censor_at_next_alarm:
            onsets = onsets.sort_values([patient_col, "datetime"])
            onsets["next_onset_t"] = onsets.groupby(patient_col)["datetime"].shift(-1)
            onsets["time_to_next_alarm_h"] = (onsets["next_onset_t"] - onsets["datetime"]).dt.total_seconds() / 3600
            onsets["prev_onset_t2"] = onsets.groupby(patient_col)["datetime"].shift(1)
            onsets["time_since_prev_alarm_h"] = (onsets["datetime"] - onsets["prev_onset_t2"]).dt.total_seconds() / 3600

        non_alarm = s[s[alarm_col] == 0].copy()
        non_alarm_by_pid = dict(non_alarm.groupby(patient_col))

        # Build paired records
        rows = []
        for _, onset in onsets.iterrows():
            pid = onset[patient_col]
            t_alarm = onset["datetime"]

            fwd_ttn = onset[ttn_col]
            fwd_event = 1
            if np.isnan(fwd_ttn) or fwd_ttn > max_forward_hours:
                fwd_ttn = max_forward_hours
                fwd_event = 0
            if (
                censor_at_next_alarm
                and not np.isnan(onset.get("time_to_next_alarm_h", np.nan))
                and onset["time_to_next_alarm_h"] < fwd_ttn
            ):
                fwd_ttn = onset["time_to_next_alarm_h"]
                fwd_event = 0

            bwd_ttl = onset[ttl_col]
            bwd_event = 1
            if np.isnan(bwd_ttl) or bwd_ttl > max_backward_hours:
                bwd_ttl = max_backward_hours
                bwd_event = 0
            if (
                censor_at_next_alarm
                and not np.isnan(onset.get("time_since_prev_alarm_h", np.nan))
                and onset["time_since_prev_alarm_h"] < bwd_ttl
            ):
                bwd_ttl = onset["time_since_prev_alarm_h"]
                bwd_event = 0

            ttn0 = onset[ttn_col]
            ttl0 = onset[ttl_col]
            raw_fwd = np.nan if (np.isnan(ttn0) or ttn0 > max_forward_hours) else ttn0
            raw_bwd = np.nan if (np.isnan(ttl0) or ttl0 > max_backward_hours) else ttl0

            if pid not in non_alarm_by_pid:
                continue
            cand = non_alarm_by_pid[pid]
            dt_h = (cand["datetime"] - t_alarm).dt.total_seconds() / 3600
            eligible = cand[(dt_h.abs() >= baseline_buffer_h) & (dt_h.abs() <= baseline_window_hours)]
            if len(eligible) == 0:
                continue

            rows.append({
                patient_col: pid,
                "alarm_time": t_alarm,
                "fwd_time": fwd_ttn,
                "fwd_event": fwd_event,
                "bwd_time": bwd_ttl,
                "bwd_event": bwd_event,
                "raw_fwd": raw_fwd,
                "raw_bwd": raw_bwd,
                "time_since_last_lab": bwd_ttl,
                "n_past_alarms": _count_past_alarms(t_alarm, pid),
                "bl_ttn_pool": eligible[ttn_col].values,
                "bl_ttl_pool": eligible[ttl_col].values,
                "bl_datetime_pool": eligible["datetime"].values,
            })

        if not rows:
            logger.warning(f"[SKIP] No valid pairs for {alarm_col}")
            continue

        pdf = pd.DataFrame(rows)
        logger.info(f"{alarm_label} -> {label}: {len(pdf)} onsets, {pdf[patient_col].nunique()} patients")

        def _time_to_nearest_alarm(dt, pid, direction="forward", _times=all_onset_times_by_pid):
            if pid not in _times:
                return np.nan
            atimes = _times[pid]
            dt64 = np.datetime64(dt, "ns") if not isinstance(dt, np.datetime64) else dt
            if direction == "forward":
                future = atimes[atimes > dt64]
                if len(future) == 0:
                    return np.nan
                return (future[0] - dt64) / np.timedelta64(1, "h")
            past = atimes[atimes < dt64]
            if len(past) == 0:
                return np.nan
            return (dt64 - past[-1]) / np.timedelta64(1, "h")

        def _draw_baseline(row, rng_, max_fwd, max_bwd):
            i = rng_.randint(len(row["bl_ttn_pool"]))
            bl_ttn = row["bl_ttn_pool"][i]
            bl_ttl = row["bl_ttl_pool"][i]
            bl_dt = row["bl_datetime_pool"][i]
            pid = row[patient_col]
            bl_npa = _count_past_alarms(bl_dt, pid)

            raw_bl_fwd = np.nan if (np.isnan(bl_ttn) or bl_ttn > max_fwd) else bl_ttn
            bl_fwd_event = 1
            if np.isnan(bl_ttn) or bl_ttn > max_fwd:
                bl_ttn = max_fwd
                bl_fwd_event = 0
            t_to_next_alarm = _time_to_nearest_alarm(bl_dt, pid, direction="forward")
            if not np.isnan(t_to_next_alarm) and t_to_next_alarm < bl_ttn:
                bl_ttn = t_to_next_alarm
                bl_fwd_event = 0

            raw_bl_bwd = np.nan if (np.isnan(bl_ttl) or bl_ttl > max_bwd) else bl_ttl
            bl_bwd_event = 1
            if np.isnan(bl_ttl) or bl_ttl > max_bwd:
                bl_ttl = max_bwd
                bl_bwd_event = 0
            t_since_prev_alarm = _time_to_nearest_alarm(bl_dt, pid, direction="backward")
            if not np.isnan(t_since_prev_alarm) and t_since_prev_alarm < bl_ttl:
                bl_ttl = t_since_prev_alarm
                bl_bwd_event = 0

            return (bl_ttn, bl_fwd_event, bl_ttl, bl_bwd_event, raw_bl_fwd, raw_bl_bwd, bl_npa)

        bl = pdf.apply(lambda r: _draw_baseline(r, rng, max_forward_hours, max_backward_hours), axis=1)
        pdf["bl_fwd_time"] = bl.apply(lambda x: x[0])
        pdf["bl_fwd_event"] = bl.apply(lambda x: x[1])
        pdf["bl_bwd_time"] = bl.apply(lambda x: x[2])
        pdf["bl_bwd_event"] = bl.apply(lambda x: x[3])
        pdf["raw_bl_fwd"] = bl.apply(lambda x: x[4])
        pdf["raw_bl_bwd"] = bl.apply(lambda x: x[5])
        pdf["bl_n_past_alarms"] = bl.apply(lambda x: x[6])

        km_fwd_alarm = kaplan_meier(pdf["fwd_time"].values, pdf["fwd_event"].values)
        km_fwd_base = kaplan_meier(pdf["bl_fwd_time"].values, pdf["bl_fwd_event"].values)

        # Log rank test (forward)
        lr_result = logrank_test(
            pdf["fwd_time"],
            pdf["bl_fwd_time"],
            event_observed_A=pdf["fwd_event"],
            event_observed_B=pdf["bl_fwd_event"],
        )
        logger.info(f"  Log-rank (forward): stat={lr_result.test_statistic:.3f}, p={lr_result.p_value:.4f}")

        # Cox PH model
        cox_alarm = pdf[["fwd_time", "fwd_event", "time_since_last_lab", "n_past_alarms", patient_col]].copy()
        cox_alarm["is_alarm"] = 1
        cox_base = pdf[["bl_fwd_time", "bl_fwd_event", "bl_bwd_time", "bl_n_past_alarms", patient_col]].copy()
        cox_base = cox_base.rename(
            columns={
                "bl_fwd_time": "fwd_time",
                "bl_fwd_event": "fwd_event",
                "bl_bwd_time": "time_since_last_lab",
                "bl_n_past_alarms": "n_past_alarms",
            }
        )
        cox_base["is_alarm"] = 0
        cox_df = pd.concat([cox_alarm, cox_base], ignore_index=True)
        cox_df = cox_df[cox_df["fwd_time"] > 0].copy()

        cox_covariates = ["is_alarm", "time_since_last_lab"]
        if cox_include_n_past_alarms:
            cox_covariates.append("n_past_alarms")
        cox_df = cox_df.dropna(subset=cox_covariates)

        cph = CoxPHFitter()
        try:
            cph.fit(
                cox_df[["fwd_time", "fwd_event", *cox_covariates, patient_col]],
                duration_col="fwd_time",
                event_col="fwd_event",
                cluster_col=patient_col,
            )
        except Exception as exc:
            logger.warning(f"  Cox model failed: {exc}")
            cph = None

        results_all[alarm_label] = {
            "pdf": pdf,
            "logrank": lr_result,
            "cox": cph,
            "km_fwd_alarm": km_fwd_alarm,
            "km_fwd_base": km_fwd_base,
        }

    return results_all


def styled_km_plot(result_dict, label, max_hours=4, figsize=(4, 3)):
    """Cumulative incidence (1 - KM) bidirectional plot for one alarm and lab pair."""
    pdf = result_dict["pdf"]
    km_fwd_alarm = result_dict["km_fwd_alarm"]
    km_fwd_base = result_dict["km_fwd_base"]
    km_bwd_alarm = kaplan_meier(pdf["bwd_time"].values, pdf["bwd_event"].values)
    km_bwd_base = kaplan_meier(pdf["bl_bwd_time"].values, pdf["bl_bwd_event"].values)

    fig, ax = plt.subplots(figsize=figsize)

    ax.step(
        km_fwd_alarm[0],
        1.0 - km_fwd_alarm[1],
        where="post",
        color=_PAL_HR["alarm"],
        lw=2,
        label=f"Alarm (n={len(pdf)})",
    )
    ax.step(km_fwd_base[0], 1.0 - km_fwd_base[1], where="post", color=_PAL_HR["baseline"], lw=2, label="Baseline")
    ax.step(-km_bwd_alarm[0], 1.0 - km_bwd_alarm[1], where="post", color=_PAL_HR["alarm"], lw=2)
    ax.step(-km_bwd_base[0], 1.0 - km_bwd_base[1], where="post", color=_PAL_HR["baseline"], lw=2)

    ax.axvline(0, color=_PAL_HR["vline"], ls="-", lw=1.5, zorder=5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-max_hours, max_hours)
    ax.set_ylabel("P(lab observed)")
    ax.set_xlabel("Hours relative to alarm onset")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()

    lr = result_dict["logrank"]
    cph = result_dict["cox"]
    logger.info(f"--- {label}: cumulative incidence ---")
    logger.info(f"  Log-rank (forward): stat={lr.test_statistic:.3f}, p={lr.p_value:.4f}")
    if cph is not None:
        hr = cph.summary.loc["is_alarm", "exp(coef)"]
        hr_lo = cph.summary.loc["is_alarm", "exp(coef) lower 95%"]
        hr_hi = cph.summary.loc["is_alarm", "exp(coef) upper 95%"]
        cox_p = cph.summary.loc["is_alarm", "p"]
        logger.info(f"  Cox HR(alarm) = {hr:.2f} [{hr_lo:.2f}-{hr_hi:.2f}], p={cox_p:.4f}")

    return fig


def styled_density_plot(result_dict, label, max_hours=4, kde_bandwidth=0.3, n_bins=40, figsize=(4, 3)):
    """Raw density (histogram + KDE) of time to lab around alarm onsets."""
    from scipy.stats import gaussian_kde

    pdf = result_dict["pdf"]
    fig, ax = plt.subplots(figsize=figsize)

    bins_fwd = np.linspace(0, max_hours, n_bins)
    bins_bwd = np.linspace(0, max_hours, n_bins)

    for vals, color, lbl, bins, sign in [
        (pdf["raw_fwd"].values, _PAL_HR["alarm"], f"Alarm (n={len(pdf)})", bins_fwd, +1),
        (pdf["raw_bl_fwd"].values, _PAL_HR["baseline"], "Baseline", bins_fwd, +1),
        (pdf["raw_bwd"].values, _PAL_HR["alarm"], None, bins_bwd, -1),
        (pdf["raw_bl_bwd"].values, _PAL_HR["baseline"], None, bins_bwd, -1),
    ]:
        v = vals[np.isfinite(vals)]
        if len(v) < 3:
            continue
        counts, edges = np.histogram(v, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        bw_h = edges[1] - edges[0]
        ax.bar(sign * centers, counts, width=bw_h * 0.9, color=color, alpha=0.25, edgecolor="white", label=lbl)
        try:
            kde = gaussian_kde(v, bw_method=kde_bandwidth)
            x = np.linspace(0, max_hours, 300)
            ax.plot(sign * x, kde(x), color=color, lw=2)
        except Exception:  # noqa: S110
            pass

    ax.axvline(0, color=_PAL_HR["vline"], ls="-", lw=1.5, zorder=5)
    ax.set_xlabel("Hours relative to alarm onset")
    ax.set_ylabel("Density")
    ax.set_xlim(-max_hours, max_hours)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def main_human_response() -> Path:
    """Generate Figure 2 panel g (``circ_alarm_lab_density``) and panel h
    (``circ_alarm_lab_km``) for circulatory alarms -> arterial lactate.

    Returns the figures directory.
    """
    _setup_fonts()
    assembled = _load_assembled()
    scores_harmonised = assembled.copy()
    if "patient_id" in scores_harmonised.columns and "patientid" not in scores_harmonised.columns:
        scores_harmonised = scores_harmonised.rename(columns={"patient_id": "patientid"})

    logger.info(f"scores_harmonised: {scores_harmonised.shape}, patients: {scores_harmonised['patientid'].nunique()}")

    figures_dir = conf.beacon.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Circulatory alarm to arterial lactate
    results_circ = analyze_alarm_km(
        scores_harmonised,
        alarm_lab_pairs=[
            {
                "alarm_col": "circ_alarm",
                "lab_col": "sample_a_Lac",
                "label": "Arterial Lactate",
                "ttn_col": "sample_a_Lac_time_to_next",
                "ttl_col": "sample_a_Lac_recency",
            }
        ],
        max_forward_hours=4,
        max_backward_hours=4,
        min_hours_since_last_alarm=8,
        ignore_first_hours=2,
        censor_at_next_alarm=True,
        baseline_buffer_minutes=0,
        baseline_window_hours=72,
        n_bootstrap=500,
        cox_include_n_past_alarms=False,
    )

    res_circ = next(iter(results_circ.values()))

    fig_km = styled_km_plot(res_circ, "CIRC -> Arterial Lactate", max_hours=4)
    _save_fig(fig_km, figures_dir / "circ_alarm_lab_km")
    plt.close(fig_km)

    fig_dens = styled_density_plot(res_circ, "CIRC -> Arterial Lactate", max_hours=4)
    _save_fig(fig_dens, figures_dir / "circ_alarm_lab_density")
    plt.close(fig_dens)

    return figures_dir


#  Figure 2 i, j, k: fairness

SCORE_LABEL_PAIRS = [
    {"score": "circ_raw_prediction", "label": "Dynamic_CircFailure_8Hours", "name": "Circulatory failure (8 h)"},
    {"score": "resp_raw_prediction", "label": "Dynamic_RespFailure_mid_8Hours", "name": "Respiratory failure (8 h)"},
]
SEX_MAP = {"M": "Male", "F": "Female"}
AGE_BINS = [0, 50, 65, 120]
AGE_LABELS = ["<50", "50\u201365", "65+"]

STRAT_CONFIGS = [
    {
        "col": "sex",
        "label": "Sex",
        "group_col": "sex_label",
        "bins": None,
        "bin_labels": None,
        "map": SEX_MAP,
        "slug": "sex",
    },
    {
        "col": "age",
        "label": "Age group",
        "group_col": "age_group",
        "bins": AGE_BINS,
        "bin_labels": AGE_LABELS,
        "map": None,
        "slug": "age",
    },
]

# Bootstrap and threshold parameters.
_FAIR_N_BOOTSTRAP = 500
_FAIR_CI_ALPHA = 0.05
_FAIR_SEED = 42
_FAIR_ROLLING_N_BOOTSTRAP = 200
_FAIR_TARGET_FPR = 0.01
_FAIR_MIN_EVENTS = 5
_FAIR_MIN_TOTAL = 20
_FAIR_WINDOW = "18D"
_FAIR_CUSUM_THRESHOLD_SIGMA = 4.0
_FAIR_FORCE = False

# Subgroup colors and ordering used by the performance bar panels (j, k).
_FAIR_SUBGROUP_COLORS = {
    "Overall": "#E2E2E2",
    "Male": "#203049",
    "Female": "#6B8ABE",
    "<50": "#BD8087",
    "50\u201365": "#AA4F59",
    "65+": "#971D2B",
}
_FAIR_SUBGROUP_ORDER = ["Overall", "Male", "Female", "<50", "50\u201365", "65+"]
_FAIR_AUROC_YLIM = (0.5, 1.0)

# Stratification display labels and colors for the disparity and CUSUM panel (i).
_FAIR_STRAT_LABELS = {
    "sex_label": "Sex",
    "age_group": "Age",
    "age_sex": "Age \u00d7 Sex",
    "Sex": "Sex",
    "Age group": "Age",
    "Age \u00d7 Sex": "Age \u00d7 Sex",
}
_FAIR_STRAT_COLORS = {
    "Sex": COLOR_PALETTE["accent_blue"],
    "Age": COLOR_PALETTE["green"]["dark"],
    "Age \u00d7 Sex": COLOR_PALETTE["purple"]["dark"],
}
_FAIR_TASK_COLOR_CYCLE = ["#203049", "#6B8ABE", "#AA4F59", "#BD8087", "#971D2B"]


def _fair_strip_prefix(metric_name):
    """Strip the leading metric prefix and separator from a metric name."""
    for sep in [" \u2013 ", " -- "]:
        if sep in metric_name:
            return metric_name.split(sep, 1)[1]
    return metric_name


def _fair_format_metric_label(mtype):
    """Convert internal metric_type string to a display label."""
    label = mtype.replace("_disparity", "").replace("_", " ").upper()
    return label.replace("TPR AT FPR", "EOD").replace("AUC", "AUROC")


def _load_fairness_assembled() -> pd.DataFrame:
    """Load the assembled df for fairness, falling back to synthetic data.

    If ``conf.beacon.parquet_path`` exists, it is used. Otherwise a synthetic
    assembled dataframe is generated via ``make_dummy_assembled`` and a clear
    warning is logged (panels i through k then render from synthetic data).
    """
    from src.fairness.testdata import make_dummy_assembled

    path = conf.beacon.parquet_path
    if path.exists():
        logger.info(f"Loading assembled data from {path}")
        assembled = pd.read_parquet(path)
    else:
        logger.warning(
            "assembled parquet not found at %s; generating SYNTHETIC assembled data. "
            "Figure 2 panels i-k will render from synthetic data, NOT real BEACON data.",
            path,
        )
        assembled = make_dummy_assembled(out_path=None)

    assembled["datetime"] = pd.to_datetime(assembled["datetime"], utc=True)
    assembled["sex_label"] = assembled["sex"].map(SEX_MAP)
    assembled["age_group"] = pd.cut(assembled["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
    return assembled


def _plot_subgroup_auroc(  # noqa: C901
    results_all_df,
    score_label_pairs,
    figures_dir,
    figsize=(5, 2),
    ylim=None,
    out_prefix="v5_perf_bars",
):
    """Per task subgroup AUROC bar chart with population composition panel.

    Produces panels j (circulatory) and k (respiratory), one figure per task,
    saved as ``{out_prefix}_{circ,resp}.*``.
    """
    n_sg = len(_FAIR_SUBGROUP_ORDER)
    x = np.arange(n_sg)
    figs = []

    for pair in score_label_pairs:
        task_name = pair["name"]
        task_slug = "circ" if "circ" in pair["score"] else "resp"

        fig, (ax_perf, ax_n) = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [2, 1.2], "wspace": 0.35},
        )
        fig.suptitle(task_name, fontweight="bold", fontsize=9, y=0.98)

        # Left: AUROC bars.
        all_vals = []
        for sg_idx, sg in enumerate(_FAIR_SUBGROUP_ORDER):
            c = _FAIR_SUBGROUP_COLORS.get(sg, "#999999")
            row = results_all_df[(results_all_df["task"] == task_name) & (results_all_df["subgroup"] == sg)]
            if row.empty or np.isnan(row.iloc[0]["auroc"]):
                continue
            r = row.iloc[0]
            val = r["auroc"]
            err_lo = max(0.0, val - r["auroc_lo"])
            err_hi = max(0.0, r["auroc_hi"] - val)
            all_vals.append(val)
            ax_perf.bar(
                sg_idx,
                val,
                0.65,
                yerr=[[err_lo], [err_hi]],
                color=c,
                alpha=1.0,
                edgecolor="black",
                linewidth=0.5,
                error_kw={"linewidth": 1.0, "capsize": 2, "capthick": 1.0},
                zorder=3,
            )

        if ylim is not None:
            ax_perf.set_ylim(*ylim)
        elif all_vals:
            y_min = max(0.0, np.floor(min(all_vals) * 20) / 20 - 0.02)
            y_max = min(1.0, max(all_vals) + 0.04)
            ax_perf.set_ylim(y_min, y_max)
        ax_perf.set_xticks(x)
        ax_perf.set_xticklabels(_FAIR_SUBGROUP_ORDER, fontsize=8)
        ax_perf.set_ylabel("AUROC", fontsize=8)
        ax_perf.spines["top"].set_visible(False)
        ax_perf.spines["right"].set_visible(False)

        # Right: population composition (two vertical stacked bars).
        strat_groups = {
            "Sex": [sg for sg in _FAIR_SUBGROUP_ORDER if sg in ("Male", "Female")],
            "Age": [sg for sg in _FAIR_SUBGROUP_ORDER if sg in ("<50", "50\u201365", "65+")],
        }
        sg_n = {}
        for sg in _FAIR_SUBGROUP_ORDER:
            if sg == "Overall":
                continue
            row = results_all_df[(results_all_df["task"] == task_name) & (results_all_df["subgroup"] == sg)]
            n_raw = row.iloc[0]["n_patients"] if not row.empty else 0
            sg_n[sg] = int(n_raw) if not (row.empty or np.isnan(n_raw)) else 0

        x_positions = list(range(len(strat_groups)))
        for x_pos, (_strat_name, sg_list) in enumerate(strat_groups.items()):
            total = sum(sg_n.get(sg, 0) for sg in sg_list)
            bottom = 0.0
            for sg in sg_list:
                n_pat = sg_n.get(sg, 0)
                c = _FAIR_SUBGROUP_COLORS.get(sg, "#999999")
                frac = n_pat / total if total > 0 else 0
                ax_n.bar(x_pos, frac, bottom=bottom, color=c, alpha=1.0, edgecolor="white", linewidth=0.8, width=0.55)
                if frac > 0.08:
                    ax_n.text(
                        x_pos,
                        bottom + frac / 2,
                        f"{sg}\nn={n_pat}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )
                bottom += frac

        total_all = sum(sg_n.get(sg, 0) for sg in strat_groups["Sex"])
        ax_n.set_ylim(0, 1)
        ax_n.set_xticks(x_positions)
        ax_n.set_xticklabels(list(strat_groups.keys()), fontsize=8)
        ax_n.set_ylabel(f"Proportion (N={total_all})", fontsize=8)
        ax_n.spines["top"].set_visible(False)
        ax_n.spines["right"].set_visible(False)

        figs.append((fig, task_slug))

    for fig_i, slug_i in figs:
        _save_fig(fig_i, figures_dir / f"{out_prefix}_{slug_i}")
        plt.close(fig_i)

    return figs


def _plot_disparity_and_cusum(
    disp_combined, score_label_pairs, figures_dir, threshold_sigma=4.0, figsize=(4.2, 3), out_prefix="v5_disp_cusum"
):
    """Combined disparity and CUSUM figure, one per (metric_type, stratification).

    Produces panel i. Files are saved as ``{out_prefix}_{metric}_{strat}.*``
    (e.g. ``v5_disp_cusum_auroc_sex``).
    """
    from src.fairness.dynamic import _estimate_cusum_params, compute_cusum

    disp = disp_combined[
        disp_combined["metric_type"].str.contains("disparity", na=False)
        & ~disp_combined["metric_type"].str.contains("alarm", na=False)
    ]
    strat_configs = sorted(disp["subgroup_config"].unique())
    metric_types = sorted(disp["metric_type"].unique())

    if not metric_types:
        logger.warning("No disparity metrics found.")
        return []

    strat_slugs = {
        "sex_label": "sex",
        "age_group": "age",
        "age_sex": "age_sex",
        "Sex": "sex",
        "Age group": "age",
        "Age \u00d7 Sex": "age_sex",
    }
    mtype_slugs = {"tpr_at_fpr_disparity": "eod", "ece_disparity": "ece", "auc_disparity": "auroc"}

    figs = []
    for mtype in metric_types:
        sub = disp[disp["metric_type"] == mtype]
        mtype_label = _fair_format_metric_label(mtype)

        for strat in strat_configs:
            strat_label = _FAIR_STRAT_LABELS.get(strat, strat)
            color = _FAIR_STRAT_COLORS.get(strat_label, "grey")

            strat_sub = sub[sub["subgroup_config"] == strat]
            if strat_sub.empty:
                continue

            fig, (ax_disp, ax_cusum) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=figsize,
                gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15},
            )

            all_wcs = sorted(strat_sub["window_center"].unique())
            mnames = sorted(strat_sub["metric_name"].unique())
            task_colors = (
                [color]
                if len(mnames) == 1
                else [_FAIR_TASK_COLOR_CYCLE[ti % len(_FAIR_TASK_COLOR_CYCLE)] for ti in range(len(mnames))]
            )

            # Top: disparity time series.
            max_h4 = 0.0
            for ti, mname in enumerate(mnames):
                tc = task_colors[ti]
                task_sub = strat_sub[strat_sub["metric_name"] == mname].sort_values("window_center")
                short = _fair_strip_prefix(mname)
                ax_disp.plot(
                    task_sub["window_center"], task_sub["value"], color=tc, linewidth=2.0, marker="o", ms=4, label=short
                )
                if task_sub["value_lo"].notna().any():
                    ax_disp.fill_between(
                        task_sub["window_center"], task_sub["value_lo"], task_sub["value_hi"], alpha=0.15, color=tc
                    )

                # Bottom: CUSUM.
                vals = task_sub["value"].values
                cusum_vals = compute_cusum(vals)
                ax_cusum.plot(
                    task_sub["window_center"].values,
                    cusum_vals,
                    color=tc,
                    linewidth=2.0,
                    marker="o",
                    ms=3,
                    label=short,
                    zorder=10,
                )

                _, _, bl_std = _estimate_cusum_params(vals)
                bl_std_safe = max(bl_std, 1e-4)
                max_h4 = max(max_h4, threshold_sigma * bl_std_safe)

            # Disparity axes formatting.
            ax_disp.axhline(0, ls="--", lw=0.6, color="grey", alpha=0.5)
            ax_disp.set_title(f"{strat_label} — {mtype_label} disparity", fontweight="bold", fontsize=10, pad=8)
            ax_disp.set_ylabel("Disparity", fontsize=10)
            ax_disp.tick_params(labelsize=10)
            ax_disp.set_ylim(0, None)
            ax_disp.set_xticks([])
            ax_disp.tick_params(axis="x", length=0)
            ax_disp.spines["top"].set_visible(False)
            ax_disp.spines["right"].set_visible(False)
            ax_disp.grid(axis="x", linestyle="--", alpha=0.5)

            # CUSUM axes formatting.
            ax_cusum.axhline(
                max_h4,
                color="#971D2B",
                linewidth=1.5,
                linestyle="--",
                alpha=1.0,
                label=f"Warning ({threshold_sigma}\u03c3)",
            )
            ax_cusum.set_ylabel("CUSUM", fontsize=10)
            ax_cusum.set_xlabel("Time", fontsize=10)
            ax_cusum.set_xticks(all_wcs)
            ax_cusum.set_xticklabels(
                [f"{w.month}/{w.day}\n{w.year}" for w in all_wcs],
                fontsize=11,
                ha="center",
                va="top",
            )
            ax_cusum.tick_params(labelsize=10)
            ax_cusum.spines["top"].set_visible(False)
            ax_cusum.spines["right"].set_visible(False)
            ax_cusum.grid(axis="x", linestyle="--", alpha=0.5)

            m_slug = mtype_slugs.get(mtype, mtype.replace("_", "-"))
            s_slug = strat_slugs.get(strat, strat.replace("_", "-"))
            figs.append((fig, f"{m_slug}_{s_slug}"))

    for fig_i, slug_i in figs:
        _save_fig(fig_i, figures_dir / f"{out_prefix}_{slug_i}")
        plt.close(fig_i)

    return figs


def _compute_disparity_combined(assembled, figures_dir) -> pd.DataFrame:
    """Compute the rolling disparity table feeding the CUSUM panel (i).

    Per stratification, roll AUROC and EOD/ECE over time, then reduce to the
    max minus min disparity. The combined table is cached at
    ``disparity_combined.parquet``.
    """
    from src.fairness.dynamic import (
        compute_disparity_over_time,
        compute_eod_calibration_over_time,
        compute_fairness_over_time,
    )

    out_path = figures_dir / "disparity_combined.parquet"
    if not _FAIR_FORCE and out_path.exists():
        logger.info(f"Loading cached disparity table from {out_path}")
        return pd.read_parquet(out_path)

    scores_df = assembled.copy()
    general_df = assembled[["patient_id", "sex", "age"]].drop_duplicates("patient_id")

    disp_parts = []
    for strat in STRAT_CONFIGS:
        slug = strat["slug"]
        sg_conf = [
            {
                "col": strat["col"],
                "label": strat["label"],
                "bins": strat["bins"],
                "bin_labels": strat["bin_labels"],
                "map": strat["map"],
            }
        ]

        rolling_auc = compute_fairness_over_time(
            scores_df=scores_df,
            general_df=general_df,
            subgroup_configs=sg_conf,
            auc_pairs=SCORE_LABEL_PAIRS,
            alarm_columns=["circ_alarm", "resp_alarm"],
            min_events=_FAIR_MIN_EVENTS,
            min_total=_FAIR_MIN_TOTAL,
            n_bootstrap=_FAIR_ROLLING_N_BOOTSTRAP,
            window=_FAIR_WINDOW,
            out_path=str(figures_dir / f"rolling_auc_{slug}.parquet"),
            force=_FAIR_FORCE,
        )
        rolling_eod = compute_eod_calibration_over_time(
            assembled,
            group_col=strat["group_col"],
            score_label_pairs=SCORE_LABEL_PAIRS,
            target_fpr=_FAIR_TARGET_FPR,
            n_bootstrap=_FAIR_ROLLING_N_BOOTSTRAP,
            min_events=_FAIR_MIN_EVENTS,
            min_total=_FAIR_MIN_TOTAL,
            window=_FAIR_WINDOW,
            out_path=str(figures_dir / f"rolling_eod_{slug}.parquet"),
            force=_FAIR_FORCE,
        )
        for prefix, src in [("eod", rolling_eod), ("auc", rolling_auc)]:
            disp_parts.append(
                compute_disparity_over_time(
                    src,
                    out_path=str(figures_dir / f"disp_{prefix}_{slug}.parquet"),
                    force=_FAIR_FORCE,
                )
            )

    disp_full = pd.concat(disp_parts, ignore_index=True)
    norm = {s["label"]: s["group_col"] for s in STRAT_CONFIGS}
    disp_full["subgroup_config"] = disp_full["subgroup_config"].map(lambda x: norm.get(x, x))
    disp_full.to_parquet(out_path, index=False)
    logger.info(f"Saved disparity table to {out_path}")
    return disp_full


def main_fairness() -> Path:
    """Generate Figure 2 panels i (``v5_disp_cusum_*``), j (``v5_perf_bars_circ``),
    and k (``v5_perf_bars_resp``).

    Computes the intermediate ``static_auroc_all.parquet`` and
    ``disparity_combined.parquet`` from the assembled dataframe (real parquet if
    present, synthetic data otherwise), then plots. Returns the figures directory.
    """
    from src.fairness.static import compute_results_all_df

    _setup_fonts()
    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 7,
    })

    assembled = _load_fairness_assembled()
    logger.info(
        "Fairness assembled: %d rows, %d patients",
        assembled.shape[0],
        assembled["patient_id"].nunique(),
    )

    figures_dir = conf.beacon.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Panels j, k: per task subgroup AUROC and composition bars
    results_all_df = compute_results_all_df(
        assembled,
        SCORE_LABEL_PAIRS,
        n_bootstrap=_FAIR_N_BOOTSTRAP,
        ci_alpha=_FAIR_CI_ALPHA,
        seed=_FAIR_SEED,
        out_path=str(figures_dir / "static_auroc_all.parquet"),
        force=_FAIR_FORCE,
    )
    _plot_subgroup_auroc(
        results_all_df,
        SCORE_LABEL_PAIRS,
        figures_dir,
        figsize=(5, 2),
        ylim=_FAIR_AUROC_YLIM,
        out_prefix="v5_perf_bars",
    )

    # Panel i: disparity dynamics + CUSUM surveillance
    disp_full = _compute_disparity_combined(assembled, figures_dir)
    logger.info(
        "Disparity: %d rows, metrics: %s",
        len(disp_full),
        sorted(disp_full["metric_type"].unique()),
    )
    _plot_disparity_and_cusum(
        disp_full,
        SCORE_LABEL_PAIRS,
        figures_dir,
        threshold_sigma=_FAIR_CUSUM_THRESHOLD_SIGMA,
        figsize=(4.2, 3),
        out_prefix="v5_disp_cusum",
    )

    return figures_dir


#  Table 1 (Bern row): cohort statistics


def _first_nonnull(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def _normalize_sex(val):
    if pd.isna(val):
        return np.nan
    t = str(val).strip().upper()
    if t in {"M", "MALE", "1", "MÄNNLICH", "MAENNLICH"}:
        return "male"
    if t in {"F", "FEMALE", "2", "W", "WEIBLICH"}:
        return "female"
    return "other"


def main_stats() -> None:  # noqa: C901
    """Print BEACON cohort statistics for Table 1 (Bern row).

    Reports patient count, sex breakdown, age median with IQR and range, MedLog
    record count, and the calendar span of active alarms.
    """
    assembled = _load_assembled()

    patient_col = "patient_id"
    if patient_col not in assembled.columns:
        msg = f"Expected column {patient_col!r} in assembled data."
        raise ValueError(msg)

    n_patients = assembled[patient_col].nunique()
    n_records = len(assembled)

    sex_candidates = [c for c in ("sex", "gender", "Sex", "Gender") if c in assembled.columns]
    sex_col = sex_candidates[0] if sex_candidates else None
    age_candidates = [c for c in ("age", "Age") if c in assembled.columns]
    age_col = age_candidates[0] if age_candidates else None

    agg_spec = {}
    if sex_col:
        agg_spec["sex"] = (sex_col, _first_nonnull)
    if age_col:
        agg_spec["age"] = (age_col, _first_nonnull)

    if agg_spec:
        per_patient = assembled.groupby(patient_col, as_index=False).agg(**{
            k: (v[0], v[1]) for k, v in agg_spec.items()
        })
    else:
        per_patient = assembled[[patient_col]].drop_duplicates()

    if sex_col and "sex" in per_patient.columns:
        per_patient["sex_norm"] = per_patient["sex"].map(_normalize_sex)
        n_male = int((per_patient["sex_norm"] == "male").sum())
        n_female = int((per_patient["sex_norm"] == "female").sum())
        n_sex_other = int((per_patient["sex_norm"] == "other").sum())
        n_sex_missing = int(per_patient["sex_norm"].isna().sum())
    else:
        n_male = n_female = n_sex_other = n_sex_missing = None

    ages = None
    if age_col and "age" in per_patient.columns:
        ages = pd.to_numeric(per_patient["age"], errors="coerce").dropna()
        age_med = float(ages.median())
        q1, q3 = float(ages.quantile(0.25)), float(ages.quantile(0.75))
        age_min, age_max = float(ages.min()), float(ages.max())

    logger.info("Cohort (BEACON assembled)")
    logger.info(f"  Patients:              {n_patients:,}")
    if sex_col:
        logger.info(f"  Male:                  {n_male:,}")
        logger.info(f"  Female:                {n_female:,}")
        if n_sex_other or n_sex_missing:
            logger.info(f"  Sex other / missing:   {n_sex_other:,} / {n_sex_missing:,}")
    else:
        logger.info("  Male / Female:         (no sex/gender column in assembled)")
    if ages is not None and len(ages) > 0:
        logger.info(
            f"  Age (years):           median {age_med:.1f} "
            f"(IQR {q1:.1f}\u2013{q3:.1f}; range {age_min:.1f}\u2013{age_max:.1f})"
        )
    else:
        logger.info("  Age:                   (no age column in assembled)")
    logger.info(f"  MedLog records (rows): {n_records:,}")

    # Alert data duration
    alarm_mask = pd.Series(False, index=assembled.index)
    for col in ("circ_alarm", "resp_alarm"):
        if col in assembled.columns:
            s = assembled[col]
            if s.dtype == bool:
                alarm_mask |= s.fillna(False)
            else:
                alarm_mask |= pd.to_numeric(s, errors="coerce").fillna(0).ne(0)

    alerts = assembled.loc[alarm_mask, "datetime"]
    if len(alerts) == 0:
        logger.info("No rows with circ_alarm or resp_alarm active - cannot compute alert duration.")
    else:
        t_min, t_max = alerts.min(), alerts.max()
        days = (t_max - t_min).total_seconds() / 86400
        logger.info(f"Alert timesteps:         {len(alerts):,}")
        logger.info(f"  First alert datetime:  {t_min}")
        logger.info(f"  Last alert datetime:   {t_max}")
        logger.info(f"  Span (max - min):      {days:.2f} days")


# Suppress numerical warnings from bootstrap and percentile on tiny subgroups.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
