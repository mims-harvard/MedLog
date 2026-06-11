"""New York pilot analysis.

Figure 5 inputs and outputs:
    roc_curve.pkl produces fig09_roc_curve (panel a)
    time_of_day.pkl produces fig04_time_of_day (panel b)
    appt_changed.pkl produces fig01_appt_changed (panel c)
    patient_called.pkl produces fig07_patient_called (panel d)
    weather_individual_days.pkl and weather_normal_reference.pkl produce fig14_weather_event_deltas (panels e and f)
    weather_results.pkl produces fig12_weather_results
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src.config import conf

logger = logging.getLogger(__name__)

# Palette.
C_BLUE = "#6B8ABE"
C_ACCENT = "#203049"

# Use Arial when available, falling back to the default font otherwise.
if any("Arial" in f.name for f in fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.bf"] = "Arial:bold"

plt.rcParams["font.size"] = 10

# Stable ordering for categorical axes.
cat_orders = {
    "Appointment_Changed": ["Yes", "No"],
    "Patient_Called": ["Yes", "No"],
    "weekday": ["Weekday", "Weekend"],
    "time_of_day": ["Morning", "Afternoon", "Evening", "Night"],
}

# Weather event days mapped to category, used by fig14.
EVENT_CATEGORIES = {
    "2025-06-19": "Severe thunderstorms",
    "2025-07-03": "Severe thunderstorms",
    "2025-07-08": "Severe thunderstorms",
    "2025-07-14": "Flash flooding",
    "2025-07-25": "Severe thunderstorms",
    "2025-07-31": "Flash flooding",
    "2025-08-13": "Severe thunderstorms",
    "2025-09-06": "Severe thunderstorms",
    "2025-09-07": "Severe thunderstorms",
    "2025-12-02": "Interior snow",
    "2025-12-13": "Winter storm",
    "2025-12-14": "Winter storm",
    "2025-12-23": "Interior snow",
    "2025-12-26": "Winter storm",
    "2025-12-27": "Winter storm",
}

C_GRAY = "#BFBFBF"
C_PURPLE = "#6B5B95"

CATEGORY_COLORS = {
    "Heat wave": "#D27F65",
    "Severe thunderstorms": C_ACCENT,
    "Flash flooding": C_BLUE,
    "Interior snow": C_PURPLE,
    "Winter storm": C_PURPLE,
    "Control": C_GRAY,
}

CATEGORY_ORDER = [
    "Heat wave",
    "Severe thunderstorms",
    "Flash flooding",
    "Interior snow",
    "Winter storm",
    "Control",
]


def _fmt_event_date(date_str: str) -> str:
    """Format a weather event date as e.g. ``July 14``."""
    return pd.to_datetime(date_str).strftime("%B %-d")


def _legend_label(cat: str) -> str:
    """Collapse the two winter categories into a single legend entry."""
    if cat in ("Interior snow", "Winter storm"):
        return "Winter storm or interior snow"
    return cat


def _save_fig(path: Path) -> None:
    """Save the current figure as PDF, SVG, and 600 dpi PNG, then close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close()


def _annotate_counts(ax, xs, ys, counts, dy=0.005):
    """Place ``n=`` labels above (x, y) points."""
    for x, y, n in zip(xs, ys, counts):
        ax.text(x, y + dy, f"$n$={int(n):,}", ha="center", va="bottom", fontsize=8)


def _plot_stratified(pkl_file, x_col, xlabel, title_suffix, plot_type, save_path):
    """Render a stratified bar, line, or auroc figure from a cached pkl and save it."""
    cache_dir = conf.mssm.cache_dir
    plot_data = pd.read_pickle(cache_dir / pkl_file)  # noqa: S301

    if x_col in cat_orders:
        plot_data[x_col] = pd.Categorical(plot_data[x_col], categories=cat_orders[x_col], ordered=True)
        plot_data = plot_data.sort_values(x_col)

    x = np.arange(len(plot_data))
    width = 0.38

    figsize = (3.25, 4) if plot_type == "bar" else (6.5, 4)
    _fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "bar":
        ax.bar(
            x - width / 2,
            plot_data["mean_prob"],
            width,
            yerr=[plot_data["prob_lower"], plot_data["prob_upper"]],
            capsize=3,
            color=C_BLUE,
            edgecolor="black",
            linewidth=0.6,
            label="Mean predicted probability",
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            plot_data["true_rate"],
            width,
            yerr=[plot_data["nsr_lower"], plot_data["nsr_upper"]],
            capsize=3,
            color=C_ACCENT,
            edgecolor="black",
            linewidth=0.6,
            label="Observed no show proportion",
            zorder=3,
        )
        ax.set_ylabel("Probability of no show", fontsize=10)
    elif plot_type == "auroc":
        ax.errorbar(
            x,
            plot_data["AUROC"],
            yerr=[plot_data["AUROC_lower"], plot_data["AUROC_upper"]],
            fmt="o-",
            capsize=3,
            color=C_BLUE,
            linewidth=1.2,
            markersize=5,
        )
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUROC", fontsize=10)
    elif plot_type == "line":
        ax.errorbar(
            x,
            plot_data["mean_prob"],
            yerr=[plot_data["prob_lower"], plot_data["prob_upper"]],
            fmt="-o",
            capsize=3,
            color=C_BLUE,
            linewidth=1.2,
            markersize=5,
            label="Mean predicted probability",
        )
        ax.errorbar(
            x,
            plot_data["true_rate"],
            yerr=[plot_data["nsr_lower"], plot_data["nsr_upper"]],
            fmt="-o",
            capsize=3,
            color=C_ACCENT,
            linewidth=1.2,
            markersize=5,
            label="Observed no show proportion",
        )
        ax.set_ylabel("Probability of no show", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_data[x_col], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if plot_type != "auroc":
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False, fontsize=9)
    plt.tight_layout()
    _save_fig(save_path)
    logger.info(f"Saved figure to {save_path}")


def plot_roc_curve(save_path: Path) -> None:
    """Panel (a): ROC curve from ``roc_curve.pkl``."""
    cache_dir = conf.mssm.cache_dir
    roc_data = pd.read_pickle(cache_dir / "roc_curve.pkl")  # noqa: S301

    _fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot(roc_data["fpr"], roc_data["tpr"], color=C_BLUE, linewidth=2, label=f"AUROC = {roc_data['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    _save_fig(save_path)
    logger.info(f"Saved figure to {save_path}")


def plot_weather_results(save_path: Path) -> None:
    """Aggregate AUROC and observed no show proportion by weather stratum from ``weather_results.pkl``."""
    cache_dir = conf.mssm.cache_dir
    weather_results = pd.read_pickle(cache_dir / "weather_results.pkl")  # noqa: S301

    _fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].bar(
        weather_results["label"],
        weather_results["auroc"],
        yerr=[weather_results["auroc_lower"].fillna(0), weather_results["auroc_upper"].fillna(0)],
        capsize=3,
        color=C_BLUE,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    axes[0].set_ylabel("AUROC", fontsize=10)

    axes[1].bar(
        weather_results["label"],
        weather_results["base_rate"],
        yerr=[weather_results["nsr_lower"].fillna(0), weather_results["nsr_upper"].fillna(0)],
        capsize=3,
        color=C_ACCENT,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    axes[1].set_ylabel("Observed no show proportion", fontsize=10)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_fig(save_path)
    logger.info(f"Saved figure to {save_path}")


def plot_weather_event_deltas(save_path: Path) -> None:
    """Panels (e) and (f): per event change in observed no show proportion and AUROC during each
    severe weather event and the heat wave month relative to the Normal Day baseline.

    Reads ``weather_individual_days.pkl``, ``weather_normal_reference.pkl``, and the heat wave
    summary row from ``weather_results.pkl``.
    """
    cache_dir = conf.mssm.cache_dir
    individual_df = pd.read_pickle(cache_dir / "weather_individual_days.pkl")  # noqa: S301
    ref = pd.read_pickle(cache_dir / "weather_normal_reference.pkl")  # noqa: S301
    normal_auroc = ref["normal_auroc"]
    normal_nsr = ref["normal_nsr"]

    events_df = individual_df[individual_df["type"] == "Weather Event"].copy()
    events_df = events_df.sort_values("date").reset_index(drop=True)
    events_df["category"] = events_df["date"].map(EVENT_CATEGORIES)
    events_df["color"] = events_df["category"].map(CATEGORY_COLORS)
    events_df["label"] = events_df["date"].map(_fmt_event_date)

    controls_df = individual_df[individual_df["type"] == "Control"].copy()
    controls_df = controls_df.sort_values("date").reset_index(drop=True)
    controls_df["category"] = "Control"
    controls_df["color"] = CATEGORY_COLORS["Control"]
    controls_df["label"] = controls_df["date"].map(_fmt_event_date)

    weather_results_full = pd.read_pickle(cache_dir / "weather_results.pkl")  # noqa: S301
    heat_wave = weather_results_full.set_index("label").loc["Heat Wave Month"]
    heat_wave_row = pd.DataFrame([
        {
            "date": pd.NA,
            "type": "Weather Event",
            "auroc": heat_wave["auroc"],
            "auroc_lower": heat_wave["auroc_lower"],
            "auroc_upper": heat_wave["auroc_upper"],
            "base_rate": heat_wave["base_rate"],
            "nsr_lower": heat_wave["nsr_lower"],
            "nsr_upper": heat_wave["nsr_upper"],
            "n": int(heat_wave["n"]),
            "category": "Heat wave",
            "color": "#D27F65",
            "label": "June (average)",
        }
    ])

    df = pd.concat([heat_wave_row, events_df, controls_df], ignore_index=True)
    df["delta_auroc"] = df["auroc"] - normal_auroc
    df["delta_nsr"] = df["base_rate"] - normal_nsr

    n_severe = 1 + len(events_df)  # heat wave + event days
    y = np.arange(len(df), dtype=float)
    y[n_severe:] += 1.0  # one row vertical gap before controls

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True, gridspec_kw={"width_ratios": [4, 4, 1.5]})

    for ax, xcol, lo_col, hi_col, xlabel in [
        (axes[0], "delta_nsr", "nsr_lower", "nsr_upper", "Δ no show proportion (event \u2212 normal)"),
        (axes[1], "delta_auroc", "auroc_lower", "auroc_upper", "Δ AUROC (event \u2212 normal)"),
    ]:
        ax.errorbar(
            df[xcol],
            y,
            xerr=[df[lo_col].fillna(0), df[hi_col].fillna(0)],
            fmt="none",
            ecolor="gray",
            elinewidth=1,
            capsize=3,
            zorder=2,
        )
        ax.scatter(df[xcol], y, s=30, c=df["color"], edgecolor="black", linewidth=0.6, zorder=3)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.8, alpha=0.8, zorder=1)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)

    axes[2].set_xticks([])
    axes[2].set_xlim(0, 1)
    for sp in ("top", "right", "left", "bottom"):
        axes[2].spines[sp].set_visible(False)
    axes[2].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axes[2].tick_params(axis="y", which="both", left=False)
    for i, (_, row) in enumerate(df.iterrows()):
        axes[2].text(0.1, y[i], f"$n$={int(row['n']):,}", ha="left", va="center", fontsize=9)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["label"], fontsize=9)
    axes[0].invert_yaxis()

    seen = set()
    legend_entries = []
    for c in CATEGORY_ORDER:
        if c not in df["category"].unique():
            continue
        label = _legend_label(c)
        if label in seen:
            continue
        legend_entries.append((label, CATEGORY_COLORS[c]))
        seen.add(label)

    patch_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.6, label=label) for label, color in legend_entries
    ]
    normal_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.8,
        alpha=0.8,
        label=f"Normal day (AUROC = {normal_auroc:.3f}, no show proportion = {normal_nsr:.3f})",
    )

    patch_legend = fig.legend(
        handles=patch_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=len(patch_handles),
        frameon=False,
        fontsize=9,
    )
    fig.add_artist(patch_legend)
    fig.legend(
        handles=[normal_handle], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False, fontsize=9
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    _save_fig(save_path)
    logger.info(f"Saved figure to {save_path}")


def main_figures() -> None:
    """Render Figure 5 panels a-f from cached aggregates (no raw PHI required)."""
    out_dir = conf.mssm.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Panel (a): ROC curve.
    plot_roc_curve(out_dir / "fig09_roc_curve.pdf")

    # Panel (b): predicted vs observed no show by time of day.
    _plot_stratified(
        pkl_file="time_of_day.pkl",
        x_col="time_of_day",
        xlabel="Time of day",
        title_suffix="Predicted vs. true no-show rate by time of day",
        plot_type="bar",
        save_path=out_dir / "fig04_time_of_day.pdf",
    )

    # Panel (c): by appointment changed.
    _plot_stratified(
        pkl_file="appt_changed.pkl",
        x_col="Appointment_Changed",
        xlabel="Appointment changed",
        title_suffix="Predicted vs. true no-show rate by appointment change",
        plot_type="bar",
        save_path=out_dir / "fig01_appt_changed.pdf",
    )

    # Panel (d): by patient called.
    _plot_stratified(
        pkl_file="patient_called.pkl",
        x_col="Patient_Called",
        xlabel="Patient called",
        title_suffix="Predicted vs. true no-show rate by patient called",
        plot_type="bar",
        save_path=out_dir / "fig07_patient_called.pdf",
    )

    # Aggregate weather summary by stratum.
    plot_weather_results(out_dir / "fig12_weather_results.pdf")

    # Panels (e) and (f): per event delta dot plot relative to the normal day baseline.
    # Rendered only when the per event weather aggregates are present.
    cache_dir = conf.mssm.cache_dir
    if (cache_dir / "weather_individual_days.pkl").exists() and (cache_dir / "weather_normal_reference.pkl").exists():
        plot_weather_event_deltas(out_dir / "fig14_weather_event_deltas.pdf")
    else:
        logger.warning(
            "Skipping fig14_weather_event_deltas: weather_individual_days.pkl / weather_normal_reference.pkl "
            "not found in %s",
            cache_dir,
        )


def _calibration_diffs(pkl_file, x_col, highlight):
    """Print per-stratum calibration gaps (observed - predicted) for a cached pkl.

    Calibration difference = observed no show proportion (``true_rate``) minus mean
    predicted probability (``mean_prob``). The paper reports the *absolute*
    difference, so ``|diff|`` (in percentage points) is shown alongside the signed
    value.
    """
    cache_dir = conf.mssm.cache_dir
    d = pd.read_pickle(cache_dir / pkl_file).copy()  # noqa: S301
    d["diff"] = d["true_rate"] - d["mean_prob"]
    logger.info(f"{pkl_file}  (observed - predicted)")
    for _, r in d.iterrows():
        star = "  <--" if r[x_col] == highlight else ""
        logger.info(
            f"  {r[x_col]!s:<10} predicted={r['mean_prob']:.3f}  observed={r['true_rate']:.3f}  "
            f"diff={r['diff']:+.3f}  |diff|={abs(r['diff']):.3f} ({abs(r['diff']) * 100:.1f} pp){star}"
        )


def main_stats() -> None:
    """Print MSSM reported statistics (calibration gaps + weather deltas) for Figure 5."""
    cache_dir = conf.mssm.cache_dir

    # Calibration differences for panels b through d.
    logger.info("Panel b - time of day (Evening: observed EXCEEDS predicted):")
    _calibration_diffs("time_of_day.pkl", "time_of_day", highlight="Evening")

    logger.info("Panel c - appointment changed after scheduling:")
    _calibration_diffs("appt_changed.pkl", "Appointment_Changed", highlight="Yes")

    logger.info("Panel d - documented outreach call (patient called):")
    _calibration_diffs("patient_called.pkl", "Patient_Called", highlight="Yes")

    # Weather statistics for panel e, all relative to the Normal Day baseline.
    wr = pd.read_pickle(cache_dir / "weather_results.pkl").set_index("label")  # noqa: S301
    normal_nsr = wr.loc["Normal Day", "base_rate"]
    normal_auroc = wr.loc["Normal Day", "auroc"]
    logger.info(f"Baseline (Normal Day): no show proportion = {normal_nsr:.3f}, AUROC = {normal_auroc:.3f}")

    # Aggregate averages: Weather Event average and Heat Wave Month average, vs. baseline.
    logger.info("Aggregate averages vs. Normal Day:")
    for label in ["Weather Event", "Heat Wave Month"]:
        nsr = wr.loc[label, "base_rate"]
        auroc = wr.loc[label, "auroc"]
        abs_inc = nsr - normal_nsr
        logger.info(
            f"  {label:<16} no show={nsr:.3f} (+{abs_inc:.3f} abs, +{abs_inc * 100:.1f} pp, "
            f"+{abs_inc / normal_nsr * 100:.1f}% rel)  AUROC={auroc:.3f} (decline {normal_auroc - auroc:.3f})"
        )


if __name__ == "__main__":
    main_figures()
    main_stats()
