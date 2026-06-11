"""Dynamic/rolling fairness analysis: computations, visualizations, and surveillance."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Dict, List, Optional

from sklearn.metrics import roc_auc_score

from .utils import (
    PALETTE, SUBGROUP_COLORS, _strip_prefix, _build_patient_index,
    cluster_bootstrap_auc, find_threshold_for_fpr,
    compute_tpr_fpr_at_threshold, compute_ece,
    _bin_continuous, _cluster_bootstrap_stat, _auc_fn, _rate_fn,
    _compute_subgroup_metric, _extract_ys_pids,
)

# Default mapping from config labels to DataFrame column names
_DEFAULT_CONFIG_TO_GROUP_VAR = {
    "Sex": "sex_label", "Age group": "age_group",
    "Age × Sex": "age_sex", "Age x Sex": "age_sex",
    "sex_label": "sex_label", "age_group": "age_group", "age_sex": "age_sex",
}

# Shared stratification display labels and colors for stacked panel plots
_STRAT_LABELS = {
    "sex_label": "Sex", "age_group": "Age", "age_sex": "Age × Sex",
    "Sex": "Sex", "Age group": "Age", "Age × Sex": "Age × Sex",
}
_STRAT_COLORS = {
    "Sex": PALETTE["accent_blue"], "Age": PALETTE["green"]["dark"],
    "Age × Sex": PALETTE["purple"]["dark"],
}


def _format_metric_label(mtype):
    """Convert internal metric_type string to display label."""
    label = mtype.replace("_disparity", "").replace("_", " ").upper()
    return label.replace("TPR AT FPR", "EOD").replace("AUC", "AUROC")


def _infer_window(datetime_series: pd.Series, n_windows: int = 6, min_days: int = 3) -> str:
    """Infer a non-overlapping rolling window string from data span."""
    n_windows = max(1, n_windows)
    span_days = (datetime_series.max() - datetime_series.min()).total_seconds() / 86400
    days = max(min_days, int(span_days / n_windows))
    return f"{days}D"


def reshape_rolling_to_static(
    rolling_auc_dfs,
    rolling_eod_dfs,
    assembled,
    score_label_pairs,
    config_label_to_group_var=None,
    n_bootstrap=200,
    ci_alpha=0.05,
    seed=42,
    target_fpr=0.01,
):
    """Extract latest window from rolling DataFrames, reshape to static format.

    Returns (results_all_dyn, eod_dyn, cal_dyn) matching the column layout
    of results_all_df, eod_results_df, cal_results_df from the static section.
    """
    if config_label_to_group_var is None:
        config_label_to_group_var = _DEFAULT_CONFIG_TO_GROUP_VAR

    task_names = [p["name"] for p in score_label_pairs]

    # --- 1. Latest window with valid data from AUROC rolling DataFrames ---
    auc_all = pd.concat(rolling_auc_dfs, ignore_index=True)
    auc_only = auc_all[auc_all["metric_type"] == "auc"]
    _valid_wcs = auc_only.dropna(subset=["value"]).groupby("window_center").size()
    if _valid_wcs.empty:
        latest_wc_auc = auc_all["window_center"].max()
    else:
        latest_wc_auc = _valid_wcs.index.max()
    latest_auc = auc_only[auc_only["window_center"] == latest_wc_auc].copy()

    results_all_rows = []
    for _, row in latest_auc.iterrows():
        task = _strip_prefix(row["metric_name"])
        if task not in task_names:
            continue
        gv = config_label_to_group_var.get(row["subgroup_config"], row["subgroup_config"])
        n_pat = row.get("n_patients", 0)
        n_pos = row.get("n_pos", 0)
        results_all_rows.append({
            "task": task, "group_var": gv, "subgroup": row["subgroup"],
            "n_patients": 0 if pd.isna(n_pat) else int(n_pat),
            "n_pos": 0 if pd.isna(n_pos) else int(n_pos),
            "prevalence": row["n_pos"] / row["n_total"] if row.get("n_total", 0) > 0 else np.nan,
            "auroc": row["value"], "auroc_lo": row["value_lo"], "auroc_hi": row["value_hi"],
        })

    # Overall AUROC from latest window's raw data
    eod_all = pd.concat(rolling_eod_dfs, ignore_index=True)
    latest_wc_eod = eod_all["window_center"].max()

    all_wcs = sorted(auc_all["window_center"].unique())
    if len(all_wcs) >= 2:
        half_win = (all_wcs[1] - all_wcs[0]) / 2  # M1: half-window is half-stride
    else:
        half_win = pd.Timedelta("15D")
    win_lo = latest_wc_auc - half_win
    win_hi = latest_wc_auc + half_win
    dt_col = "datetime"
    win_assembled = assembled[
        (pd.to_datetime(assembled[dt_col]) >= win_lo) &
        (pd.to_datetime(assembled[dt_col]) < win_hi)
    ]

    for pair in score_label_pairs:
        sc, lc, name = pair["score"], pair["label"], pair["name"]
        y = win_assembled[lc].astype(float).values
        s = win_assembled[sc].astype(float).values
        pids = win_assembled["patient_id"].values
        mask = np.isfinite(y) & np.isfinite(s)
        y, s, pids = y[mask].astype(int), s[mask], pids[mask]
        if y.sum() >= 5 and (len(y) - y.sum()) >= 5:
            auc_val, lo, hi = cluster_bootstrap_auc(y, s, pids, n_bootstrap, ci_alpha, seed)
        else:
            auc_val = lo = hi = np.nan
        results_all_rows.append({
            "task": name, "group_var": "overall", "subgroup": "Overall",
            "n_patients": len(np.unique(pids)), "n_pos": int(y.sum()),
            "prevalence": y.sum() / len(y) if len(y) > 0 else np.nan,
            "auroc": auc_val, "auroc_lo": lo, "auroc_hi": hi,
        })

    results_all_dyn = pd.DataFrame(results_all_rows)

    # --- 2. Latest window with valid EOD (TPR at fixed FPR) ---
    _valid_eod_wcs = eod_all[
        (eod_all["metric_type"] == "tpr_at_fpr")
    ].dropna(subset=["value"]).groupby("window_center").size()
    if not _valid_eod_wcs.empty:
        latest_wc_eod = _valid_eod_wcs.index.max()
    latest_eod = eod_all[
        (eod_all["window_center"] == latest_wc_eod) & (eod_all["metric_type"] == "tpr_at_fpr")
    ].copy()

    eod_rows = []
    for _, row in latest_eod.iterrows():
        task = _strip_prefix(row["metric_name"])
        if task not in task_names:
            continue
        gv = config_label_to_group_var.get(row["subgroup_config"], row["subgroup_config"])
        eod_rows.append({
            "task": task, "group_var": gv, "subgroup": row["subgroup"],
            "tpr": row["value"], "tpr_lo": row["value_lo"], "tpr_hi": row["value_hi"],
            "fpr": np.nan,
            "global_threshold": np.nan,
            "n_patients": row.get("n_patients", 0), "n_pos": row.get("n_pos", 0),
        })

    # Recompute FPR per subgroup from latest window raw data
    for pair in score_label_pairs:
        sc, lc, name = pair["score"], pair["label"], pair["name"]
        y_w = win_assembled[lc].astype(float).values
        s_w = win_assembled[sc].astype(float).values
        valid = np.isfinite(y_w) & np.isfinite(s_w)
        y_w, s_w = y_w[valid].astype(int), s_w[valid]
        thr = find_threshold_for_fpr(y_w, s_w, target_fpr) if y_w.sum() >= 5 else np.nan

        for erow in eod_rows:
            if erow["task"] != name:
                continue
            erow["global_threshold"] = thr
            gv = erow["group_var"]
            sg = erow["subgroup"]
            if gv in win_assembled.columns:
                available = set(win_assembled[gv].astype(str).unique())
                if sg not in available:
                    warnings.warn(
                        f"Subgroup '{sg}' not found in column '{gv}'; FPR will be NaN",
                        stacklevel=2,
                    )
                    continue
                grp = win_assembled[win_assembled[gv].astype(str) == sg]
            else:
                continue
            yg = grp[lc].astype(float).values
            sg_scores = grp[sc].astype(float).values
            m = np.isfinite(yg) & np.isfinite(sg_scores)
            yg, sg_scores = yg[m].astype(int), sg_scores[m]
            if len(yg) > 0 and not np.isnan(thr):
                _, fpr_val = compute_tpr_fpr_at_threshold(yg, sg_scores, thr)
                erow["fpr"] = fpr_val

    eod_dyn = pd.DataFrame(eod_rows)

    # --- 3. Latest window calibration (ECE) ---
    _valid_cal_wcs = eod_all[
        (eod_all["metric_type"] == "ece")
    ].dropna(subset=["value"]).groupby("window_center").size()
    latest_wc_cal = _valid_cal_wcs.index.max() if not _valid_cal_wcs.empty else latest_wc_eod
    latest_cal = eod_all[
        (eod_all["window_center"] == latest_wc_cal) & (eod_all["metric_type"] == "ece")
    ].copy()

    cal_rows = []
    for _, row in latest_cal.iterrows():
        task = _strip_prefix(row["metric_name"])
        if task not in task_names:
            continue
        gv = config_label_to_group_var.get(row["subgroup_config"], row["subgroup_config"])
        cal_rows.append({
            "task": task, "group_var": gv, "subgroup": row["subgroup"],
            "ece": row["value"], "ece_lo": row["value_lo"], "ece_hi": row["value_hi"],
            "n_patients": row.get("n_patients", 0),
        })

    cal_dyn = pd.DataFrame(cal_rows)

    return results_all_dyn, eod_dyn, cal_dyn


def plot_disparity_trajectory(disp_combined, score_label_pairs=None,
                              figsize_per_panel=(6, 5.5)):
    """Phase portrait: co-evolution of AUROC disparity and EOD over time."""
    if score_label_pairs is None:
        raise ValueError("score_label_pairs is required")
    task_names = [p["name"] for p in score_label_pairs]

    # Filter to fairness disparity only (exclude alarm_rate) — m7: unified filter
    disp_combined = disp_combined[
        disp_combined["metric_type"].str.contains("disparity", na=False) &
        ~disp_combined["metric_type"].str.contains("alarm", na=False)
    ]
    strat_configs = sorted(disp_combined["subgroup_config"].unique())

    n_panels = len(strat_configs)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
                             squeeze=False)

    strat_colors = {
        "Sex": PALETTE["accent_blue"],
        "Age group": PALETTE["green"]["dark"],
        "Age × Sex": PALETTE["purple"]["dark"],
        "sex_label": PALETTE["accent_blue"],
        "age_group": PALETTE["green"]["dark"],
        "age_sex": PALETTE["purple"]["dark"],
    }

    cmap = plt.cm.viridis

    for ax_idx, strat in enumerate(strat_configs):
        ax = axes[0, ax_idx]
        sub = disp_combined[disp_combined["subgroup_config"] == strat].copy()

        for t_idx, task_name in enumerate(task_names):
            auc_disp = sub[sub["metric_type"] == "auc_disparity"]
            auc_disp = auc_disp[auc_disp["metric_name"].str.contains(task_name.split("(")[0].strip(), regex=False)]
            auc_disp = auc_disp.sort_values("window_center")

            tpr_disp = sub[sub["metric_type"] == "tpr_at_fpr_disparity"]
            tpr_disp = tpr_disp[tpr_disp["metric_name"].str.contains(task_name.split("(")[0].strip(), regex=False)]
            tpr_disp = tpr_disp.sort_values("window_center")

            merged = auc_disp[["window_center", "value"]].rename(columns={"value": "auc_disp"})
            merged = merged.merge(
                tpr_disp[["window_center", "value"]].rename(columns={"value": "tpr_disp"}),
                on="window_center", how="inner",
            )
            merged = merged.dropna(subset=["auc_disp", "tpr_disp"])

            if merged.empty:
                continue

            if len(merged) == 1:
                ax.annotate(
                    "single window",
                    xy=(merged.iloc[0]["auc_disp"], merged.iloc[0]["tpr_disp"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=6, color="grey", style="italic",
                )

            n_pts = len(merged)
            colors = [cmap(i / max(1, n_pts - 1)) for i in range(n_pts)]

            for i, (_, row) in enumerate(merged.iterrows()):
                ax.scatter(row["auc_disp"], row["tpr_disp"], color=colors[i],
                           s=40, zorder=5, edgecolors="white", linewidths=0.5,
                           marker="o" if t_idx == 0 else "s")

            for i in range(len(merged) - 1):
                r0 = merged.iloc[i]
                r1 = merged.iloc[i + 1]
                ax.annotate("", xy=(r1["auc_disp"], r1["tpr_disp"]),
                            xytext=(r0["auc_disp"], r0["tpr_disp"]),
                            arrowprops=dict(arrowstyle="->", color=colors[i],
                                            lw=1.2, alpha=0.7))

            ax.scatter([], [], color="grey",
                       marker="o" if t_idx == 0 else "s",
                       label=task_name, s=40)

        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.set_xlabel("AUROC disparity (max \u2212 min)")
        ax.set_ylabel("TPR disparity / EOD (max \u2212 min)")
        strat_label = {"sex_label": "Sex", "age_group": "Age", "age_sex": "Age \u00d7 Sex"}.get(strat, strat)
        ax.set_title(f"Fairness trajectory \u2014 {strat_label}", fontweight="bold", pad=8)
        ax.legend(loc="upper right", fontsize=7)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.08)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Early", "Recent"])
        cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


def plot_radar_evolution(rolling_auc_dfs, rolling_eod_dfs,
                        score_label_pairs=None, n_snapshots=3,
                        group_var="sex_label",
                        config_label_to_group_var=None,
                        figsize=(18, 6)):
    """Small-multiples radar: fairness fingerprint at n_snapshots time points."""
    if score_label_pairs is None:
        raise ValueError("score_label_pairs is required")
    if config_label_to_group_var is None:
        config_label_to_group_var = _DEFAULT_CONFIG_TO_GROUP_VAR

    tasks = [p["name"] for p in score_label_pairs]
    short_names = {t: t.replace(" failure", "").replace(" (8 h)", "") for t in tasks}

    metric_labels = []
    for t in tasks:
        sn = short_names[t]
        metric_labels.extend([f"AUROC\n{sn}", f"TPR@1%FPR\n{sn}", f"Calibration\n{sn}"])
    n_metrics = len(metric_labels)

    # Gather window centers that have at least one non-NaN value
    auc_all = pd.concat(rolling_auc_dfs, ignore_index=True)
    eod_all = pd.concat(rolling_eod_dfs, ignore_index=True)
    valid_auc_wcs = set(auc_all.dropna(subset=["value"])["window_center"].unique())
    valid_eod_wcs = set(eod_all.dropna(subset=["value"])["window_center"].unique())
    valid_wcs = valid_auc_wcs | valid_eod_wcs
    all_wcs = sorted(valid_wcs) if valid_wcs else sorted(auc_all["window_center"].unique())

    if len(all_wcs) < n_snapshots:
        n_snapshots = len(all_wcs)
    if n_snapshots == 0:
        print("No time windows available for radar evolution.")
        return None

    indices = np.linspace(0, len(all_wcs) - 1, n_snapshots, dtype=int)
    snapshot_wcs = [all_wcs[i] for i in indices]

    # Determine subgroups for the chosen group_var
    gv_match_auc = [k for k, v in config_label_to_group_var.items()
                    if v == group_var and k in auc_all["subgroup_config"].unique()]
    gv_match_eod = [k for k, v in config_label_to_group_var.items()
                    if v == group_var and k in eod_all["subgroup_config"].unique()]
    if not gv_match_auc and not gv_match_eod:
        print(f"No data for group_var={group_var}")
        return None
    auc_config = gv_match_auc[0] if gv_match_auc else None
    eod_config = gv_match_eod[0] if gv_match_eod else None

    subgroups_auc = sorted(auc_all[auc_all["subgroup_config"] == auc_config]["subgroup"].unique()) if auc_config else []
    subgroups_eod = sorted(eod_all[eod_all["subgroup_config"] == eod_config]["subgroup"].unique()) if eod_config else []
    subgroups = sorted(set(subgroups_auc) | set(subgroups_eod))

    sg_colors_map = {
        "sex_label": [PALETTE["accent_blue"], PALETTE["red"]["dark"]],
        "age_group": [PALETTE["green"]["dark"], PALETTE["purple"]["dark"], PALETTE["yellow"]["dark"]],
    }
    sg_colors = sg_colors_map.get(group_var, SUBGROUP_COLORS[:len(subgroups)])

    # C2: Pre-build lookup dicts to replace O(N²) DataFrame scans inside _get_values
    auc_lookup = {}
    for _, row in auc_all.iterrows():
        key = (row["window_center"], row["subgroup_config"], row["subgroup"],
               row["metric_type"], _strip_prefix(str(row["metric_name"])))
        v = row["value"]
        auc_lookup[key] = v if not (isinstance(v, float) and np.isnan(v)) else np.nan

    eod_lookup = {}
    for _, row in eod_all.iterrows():
        key = (row["window_center"], row["subgroup_config"], row["subgroup"],
               row["metric_type"], _strip_prefix(str(row["metric_name"])))
        v = row["value"]
        eod_lookup[key] = v if not (isinstance(v, float) and np.isnan(v)) else np.nan

    def _get_values(wc, subgroup):
        vals = []
        for t in tasks:
            if auc_config:
                v = auc_lookup.get((wc, auc_config, subgroup, "auc", t), np.nan)
            else:
                v = np.nan
            vals.append(v)
            if eod_config:
                v = eod_lookup.get((wc, eod_config, subgroup, "tpr_at_fpr", t), np.nan)
            else:
                v = np.nan
            vals.append(v)
            if eod_config:
                ece = eod_lookup.get((wc, eod_config, subgroup, "ece", t), np.nan)
                vals.append(1.0 - ece if not (isinstance(ece, float) and np.isnan(ece)) else np.nan)
            else:
                vals.append(np.nan)
        return vals

    # Gather all values for axis scaling
    all_vals = []
    for wc in snapshot_wcs:
        for sg in subgroups:
            all_vals.extend([v for v in _get_values(wc, sg) if not np.isnan(v)])

    if not all_vals:
        print("No valid values for radar evolution.")
        return None

    val_min, val_max = min(all_vals), max(all_vals)
    margin = max(0.02, (val_max - val_min) * 0.3)
    r_min = max(0, round(val_min - margin, 2))
    r_max = min(1.0, round(val_max + margin, 2))  # M3: symmetric margin
    if r_max - r_min < 0.05:
        r_min = max(0, r_max - 0.1)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=figsize)
    snapshot_labels = ["Early", "Middle", "Recent"] if n_snapshots == 3 else [f"T{i+1}" for i in range(n_snapshots)]

    for snap_idx, wc in enumerate(snapshot_wcs):
        ax = fig.add_subplot(1, n_snapshots, snap_idx + 1, polar=True)
        wc_str = pd.Timestamp(wc).strftime("%Y-%m-%d")
        ax.set_title(f"{snapshot_labels[snap_idx]}\n({wc_str})", fontweight="bold", pad=15, fontsize=10)

        for sg_idx, sg in enumerate(subgroups):
            vals = _get_values(wc, sg)
            c = sg_colors[sg_idx % len(sg_colors)]
            # M2: keep NaN as-is; matplotlib skips missing segments rather than snapping to r_min
            vals_closed = vals + vals[:1]
            ax.plot(angles_closed, vals_closed, color=c, linewidth=1.5, label=sg)
            ax.fill(angles_closed, vals_closed, color=c, alpha=0.12)

        ax.set_ylim(r_min, r_max)
        ax.set_xticks(angles)
        ax.set_xticklabels(metric_labels, fontsize=6.5)
        n_rings = 4
        ring_vals = np.linspace(r_min, r_max, n_rings + 1)[1:]
        ax.set_yticks(ring_vals)
        ax.set_yticklabels([f"{v:.2f}" for v in ring_vals], fontsize=6, color="grey")

        if snap_idx == n_snapshots - 1:
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)

    gv_label = {"sex_label": "Sex", "age_group": "Age"}.get(group_var, group_var)
    fig.suptitle(f"Fairness fingerprint evolution \u2014 {gv_label}", fontsize=12, y=1.04)
    fig.subplots_adjust(wspace=0.5)
    return fig


def _estimate_cusum_params(values):
    """Estimate CUSUM baseline, allowance *k*, and baseline std from first ⌊N/3⌋ steps.

    Returns ``(baseline, k, bl_std)``.  Falls back to ``(0.0, 1e-4, 1e-4)``
    if the baseline block has <2 valid observations (avoids future-data leakage).
    Uses ``ddof=1`` (Bessel-corrected) sample standard deviation.
    """
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    n_total = len(values)
    bl_end = max(3, n_total // 3)
    bl_idx = np.where(valid[:bl_end])[0]
    if len(bl_idx) >= 2:
        bl_vals = values[bl_idx]
        baseline = float(np.mean(bl_vals))
        bl_std = float(np.std(bl_vals, ddof=1))
        k = max(1e-4, bl_std * 0.5)
    else:
        baseline = 0.0
        bl_std = 1e-4
        k = 1e-4
        warnings.warn(
            f"compute_cusum: first {bl_end} time steps have <2 valid values; "
            f"using baseline=0.0 (no future data used)",
            stacklevel=2,
        )
    return baseline, k, bl_std


def compute_cusum(values, baseline=None, k=None, max_nan_gap=2):
    """One-sided upper CUSUM (Page's) for detecting increases in a disparity metric.

    Parameters
    ----------
    values : array-like
        Time series of disparity values (chronologically ordered).
    baseline : float, optional
        In-control mean (μ₀). If None, estimated from the first ⌊N_total/3⌋
        time steps (temporal block), filtering NaN within that block.
        Falls back to 0.0 if the block has <2 valid observations
        (no future-data leakage).
    k : float, optional
        Allowance (slack) parameter. If None, set to 0.5 × σ(baseline period)
        using Bessel-corrected (ddof=1) sample std, targeting detection of a
        1-σ shift. Floored at 1e-4.
    max_nan_gap : int
        Number of consecutive NaN values to carry forward (hold-last). After
        more than `max_nan_gap` consecutive NaN, CUSUM resets to 0 to avoid
        stale alarms across substantial data gaps. Default 2.

    Returns
    -------
    cusum : ndarray
        Accumulated CUSUM values.

    Notes
    -----
    Recurrence: S₀ = max(0, x₀ − μ₀ − k),  Sₙ = max(0, Sₙ₋₁ + xₙ − μ₀ − k).
    """
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    if valid.sum() < 3:
        return np.full_like(values, np.nan)

    n_total = len(values)

    # Auto-estimate baseline and allowance only when not supplied by caller.
    if baseline is None or k is None:
        est_baseline, est_k, _ = _estimate_cusum_params(values)
        if baseline is None:
            baseline = est_baseline
        if k is None:
            k = est_k

    cusum = np.zeros(n_total)
    if not np.isnan(values[0]):
        cusum[0] = max(0, values[0] - baseline - k)

    # C4: track consecutive NaN run; reset after max_nan_gap consecutive NaN.
    nan_run = 1 if np.isnan(values[0]) else 0
    for i in range(1, n_total):
        if np.isnan(values[i]):
            nan_run += 1
            if nan_run > max_nan_gap:
                cusum[i] = 0.0   # reset: stale alarm after sustained gap
            else:
                cusum[i] = cusum[i - 1]   # carry-forward for short gaps
        else:
            nan_run = 0
            cusum[i] = max(0, cusum[i - 1] + (values[i] - baseline - k))
    return cusum


def plot_disparity_stacked(disp_combined, score_label_pairs=None,
                           figsize=(5, 5)):
    """Disparity time series — stacked subplots, one per stratification.

    Same visual style as plot_cusum_surveillance: vertical stack, per-strat
    color, shared x-axis.  X-ticks are aligned to window centers.
    """
    if score_label_pairs is None:
        raise ValueError("score_label_pairs is required")

    disp = disp_combined[
        disp_combined["metric_type"].str.contains("disparity", na=False) &
        ~disp_combined["metric_type"].str.contains("alarm", na=False)
    ]
    strat_configs = sorted(disp["subgroup_config"].unique())
    metric_types = sorted(disp["metric_type"].unique())

    if not metric_types:
        print("No disparity metrics found.")
        return None

    all_wcs = sorted(disp["window_center"].unique())
    n_panels = len(metric_types) * len(strat_configs)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=figsize,
        sharex=True, squeeze=False,
    )

    panel = 0
    for mtype in metric_types:
        sub = disp[disp["metric_type"] == mtype]
        mtype_label = _format_metric_label(mtype)

        for strat in strat_configs:
            ax = axes[panel, 0]
            strat_label = _STRAT_LABELS.get(strat, strat)
            color = _STRAT_COLORS.get(strat_label, "grey")

            strat_sub = sub[sub["subgroup_config"] == strat]
            if strat_sub.empty:
                ax.set_visible(False)
                panel += 1
                continue

            mnames = sorted(strat_sub["metric_name"].unique())
            task_colors = ([color] if len(mnames) == 1
                           else [SUBGROUP_COLORS[ti % len(SUBGROUP_COLORS)]
                                 for ti in range(len(mnames))])
            for ti, mname in enumerate(mnames):
                tc = task_colors[ti]
                task_sub = strat_sub[strat_sub["metric_name"] == mname].sort_values("window_center")
                short = _strip_prefix(mname)
                ax.plot(task_sub["window_center"], task_sub["value"],
                        color=tc, linewidth=2.0, marker="o", ms=4,
                        label=short)
                if task_sub["value_lo"].notna().any():
                    ax.fill_between(task_sub["window_center"],
                                    task_sub["value_lo"], task_sub["value_hi"],
                                    alpha=0.15, color=tc)

            ax.axhline(0, ls="--", lw=0.6, color="grey", alpha=0.5)
            ax.set_title(f"{strat_label} — {mtype_label} disparity (max − min)",
                         fontweight="bold", fontsize=9, pad=8)
            ax.set_ylabel("Disparity\n(max − min)", fontsize=8)
            ax.set_ylim(0, None)
            ax.legend(loc="upper left", fontsize=7)
            ax.set_xticks(all_wcs)
            ax.tick_params(labelsize=7)
            ax.tick_params(axis="x", rotation=30)
            panel += 1

    axes[-1, 0].set_xlabel("Time")
    fig.tight_layout()
    return fig


def plot_cusum_surveillance(disp_combined, score_label_pairs=None,
                             threshold_sigma=4.0, figsize=(5, 5)):
    """CUSUM fairness surveillance chart — stacked subplots per stratification.

    Each metric type produces a figure with vertically stacked panels
    (e.g. Sex, then Age). Shows warning (threshold_sigma × σ) and
    alert ((threshold_sigma + 1) × σ) threshold lines.
    """
    if score_label_pairs is None:
        raise ValueError("score_label_pairs is required")

    strat_configs = sorted(disp_combined["subgroup_config"].unique())
    metric_types = sorted(disp_combined["metric_type"].unique())
    metric_types = [m for m in metric_types if "disparity" in m and "alarm" not in m]

    if not metric_types:
        print("No disparity metrics found.")
        return None

    all_wcs = sorted(disp_combined["window_center"].unique())
    n_panels = len(metric_types) * len(strat_configs)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=figsize,
        sharex=True, squeeze=False,
    )

    panel = 0
    for mtype in metric_types:
        sub = disp_combined[disp_combined["metric_type"] == mtype]
        mtype_label = _format_metric_label(mtype)

        for strat in strat_configs:
            ax = axes[panel, 0]
            strat_label = _STRAT_LABELS.get(strat, strat)
            color = _STRAT_COLORS.get(strat_label, "grey")

            strat_sub = sub[sub["subgroup_config"] == strat]
            if strat_sub.empty:
                ax.set_visible(False)
                panel += 1
                continue

            # Iterate per task (metric_name) to avoid interleaving values
            mnames = sorted(strat_sub["metric_name"].unique())
            task_colors = ([color] if len(mnames) == 1
                           else [SUBGROUP_COLORS[ti % len(SUBGROUP_COLORS)]
                                 for ti in range(len(mnames))])
            max_h4 = max_h5 = 0.0
            for ti, mname in enumerate(mnames):
                tc = task_colors[ti]
                task_sub = strat_sub[strat_sub["metric_name"] == mname].sort_values("window_center")
                vals = task_sub["value"].values
                cusum = compute_cusum(vals)
                short = _strip_prefix(mname)
                ax.plot(task_sub["window_center"].values, cusum,
                        color=tc, linewidth=2.0, marker="o", ms=3,
                        label=short)

                # Threshold from baseline std (same estimator as compute_cusum)
                _, _, bl_std = _estimate_cusum_params(vals)
                bl_std_safe = max(bl_std, 1e-4)
                max_h4 = max(max_h4, threshold_sigma * bl_std_safe)
                max_h5 = max(max_h5, (threshold_sigma + 1) * bl_std_safe)

            ax.axhline(max_h4, color="grey", linewidth=1.5, linestyle="--",
                       alpha=0.5, label=f"Warning ({threshold_sigma}σ)")
            ax.axhline(max_h5, color="grey", linewidth=2.5, linestyle="--",
                       alpha=0.7, label=f"Alert ({threshold_sigma + 1}σ)")

            ax.set_title(f"{strat_label} — CUSUM on {mtype_label} disparity",
                         fontweight="bold", fontsize=9, pad=8)
            ax.set_ylabel("CUSUM", fontsize=8)
            ax.legend(loc="upper left", fontsize=7)
            ax.set_xticks(all_wcs)
            ax.tick_params(labelsize=7)
            ax.tick_params(axis="x", rotation=30)
            panel += 1

    axes[-1, 0].set_xlabel("Time")
    fig.tight_layout()
    return fig


def plot_disparity_and_cusum(
    disp_combined: pd.DataFrame,
    score_label_pairs: Optional[list] = None,
    threshold_sigma: float = 4.0,
    figsize: tuple = (5, 6),
    out_dir: Optional[str] = None,
    out_prefix: str = "",
    out_formats: tuple = ("png", "pdf"),
    dpi: int = 300,
) -> list:
    """Combined disparity + CUSUM figure — one per (metric_type, stratification).

    Each figure has two rows sharing an x-axis:
      - Top: disparity time series with CI bands
      - Bottom: Page's CUSUM on the same disparity series

    Returns a list of (fig, slug) tuples.
    """
    if score_label_pairs is None:
        raise ValueError("score_label_pairs is required")

    disp = disp_combined[
        disp_combined["metric_type"].str.contains("disparity", na=False) &
        ~disp_combined["metric_type"].str.contains("alarm", na=False)
    ]
    strat_configs = sorted(disp["subgroup_config"].unique())
    metric_types = sorted(disp["metric_type"].unique())

    if not metric_types:
        print("No disparity metrics found.")
        return []

    _strat_slugs = {"sex_label": "sex", "age_group": "age", "age_sex": "age_sex",
                     "Sex": "sex", "Age group": "age", "Age × Sex": "age_sex"}
    _mtype_slugs = {"tpr_at_fpr_disparity": "eod", "ece_disparity": "ece",
                     "auc_disparity": "auroc"}

    figs = []
    for mtype in metric_types:
        sub = disp[disp["metric_type"] == mtype]
        mtype_label = _format_metric_label(mtype)

        for strat in strat_configs:
            strat_label = _STRAT_LABELS.get(strat, strat)
            color = _STRAT_COLORS.get(strat_label, "grey")

            strat_sub = sub[sub["subgroup_config"] == strat]
            if strat_sub.empty:
                continue

            fig, (ax_disp, ax_cusum) = plt.subplots(
                2, 1, sharex=True, figsize=figsize,
                gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15},
            )

            all_wcs = sorted(strat_sub["window_center"].unique())
            mnames = sorted(strat_sub["metric_name"].unique())
            task_colors = ([color] if len(mnames) == 1
                           else [SUBGROUP_COLORS[ti % len(SUBGROUP_COLORS)]
                                 for ti in range(len(mnames))])

            # ── Top: disparity time series ──────────────────────────────────
            max_h4 = max_h5 = 0.0
            for ti, mname in enumerate(mnames):
                tc = task_colors[ti]
                task_sub = strat_sub[strat_sub["metric_name"] == mname].sort_values("window_center")
                short = _strip_prefix(mname)
                ax_disp.plot(task_sub["window_center"], task_sub["value"],
                             color=tc, linewidth=2.0, marker="o", ms=4, label=short)
                if task_sub["value_lo"].notna().any():
                    ax_disp.fill_between(task_sub["window_center"],
                                         task_sub["value_lo"], task_sub["value_hi"],
                                         alpha=0.15, color=tc)

                # ── Bottom: CUSUM ───────────────────────────────────────────
                vals = task_sub["value"].values
                cusum = compute_cusum(vals)
                ax_cusum.plot(task_sub["window_center"].values, cusum,
                              color=tc, linewidth=2.0, marker="o", ms=3, label=short)

                # Threshold from baseline std (same estimator as compute_cusum)
                _, _, bl_std = _estimate_cusum_params(vals)
                bl_std_safe = max(bl_std, 1e-4)
                max_h4 = max(max_h4, threshold_sigma * bl_std_safe)
                max_h5 = max(max_h5, (threshold_sigma + 1) * bl_std_safe)

            # Disparity axes formatting
            ax_disp.axhline(0, ls="--", lw=0.6, color="grey", alpha=0.5)
            ax_disp.set_title(f"{strat_label} — {mtype_label} disparity",
                              fontweight="bold", fontsize=9, pad=8)
            ax_disp.set_ylabel("Disparity\n(max − min)", fontsize=8)
            ax_disp.set_ylim(0, None)
            ax_disp.legend(loc="upper left", fontsize=7)
            ax_disp.set_xticks(all_wcs)
            ax_disp.tick_params(labelsize=7)

            # CUSUM axes formatting
            ax_cusum.axhline(max_h4, color="grey", linewidth=1.5, linestyle="--",
                             alpha=0.5, label=f"Warning ({threshold_sigma}σ)")
            ax_cusum.axhline(max_h5, color="grey", linewidth=2.5, linestyle="--",
                             alpha=0.7, label=f"Alert ({threshold_sigma + 1}σ)")
            ax_cusum.set_ylabel("CUSUM", fontsize=8)
            ax_cusum.set_xlabel("Time")
            ax_cusum.legend(loc="upper left", fontsize=7)
            ax_cusum.set_xticks(all_wcs)
            ax_cusum.tick_params(labelsize=7)
            ax_cusum.tick_params(axis="x", rotation=30)

            fig.tight_layout()
            m_slug = _mtype_slugs.get(mtype, mtype.replace("_", "-"))
            s_slug = _strat_slugs.get(strat, strat.replace("_", "-"))
            figs.append((fig, f"{m_slug}_{s_slug}"))

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pfx = f"{out_prefix}_" if out_prefix else ""
        for fig_i, slug_i in figs:
            for fmt in out_formats:
                fig_i.savefig(Path(out_dir) / f"{pfx}{slug_i}.{fmt}",
                              dpi=dpi, bbox_inches="tight")

    return figs


# ── Chugg et al. 2023: Betting-based fairness surveillance ───────────────────
# Reference: "Auditing Fairness by Betting", NeurIPS 2023 (Spotlight)
# arXiv:2305.17570 | https://github.com/bchugg/auditing-fairness

def _test_by_betting_ons(seq1, seq2, alpha=0.05):
    """Two-sided sequential betting test H0: E[seq1_t − seq2_t] = 0 (Chugg et al. 2023).

    ONS (Online Newton Step) betting strategy verbatim from Chugg et al.
    methods.py, variable names preserved. Lambda is clamped to [−0.5, 0.5],
    so ONS adapts to detect deviations in either direction.

    Modifications from reference:
    * NaN pairs skipped (wealth held constant) for sparse clinical windows.
    * Early-exit removed; full wealth trajectory returned for visualisation.
    * Randomised stopping rule (Ville's trick, reference line 42-43) omitted.
      This makes the test slightly conservative (type-I error ≤ alpha, not
      exactly alpha at termination). For surveillance dashboards where the
      interest is the trajectory shape, this is acceptable.

    Parameters
    ----------
    seq1, seq2 : array-like
        Per-window metric values for two subgroups, same temporal alignment.
        NaN pairs are filtered before running.
    alpha : float
        Significance level; alarm threshold = 1/alpha.

    Returns
    -------
    wealth_hist : list[float]
        Wealth at each valid (non-NaN) time step.
    rejected : bool
        True if wealth exceeded 1/alpha at any point.
    """
    seq1 = np.asarray(seq1, dtype=float)
    seq2 = np.asarray(seq2, dtype=float)
    valid = ~np.isnan(seq1) & ~np.isnan(seq2)
    seq1, seq2 = seq1[valid], seq2[valid]

    if len(seq1) < 1:
        return [1.0], False

    # ── verbatim from Chugg et al. methods.py (variable names preserved) ──
    # Note: loop starts at t=1 (faithful to reference). At t=0, lambd=0 so
    # wealth is trivially 1.0; the lambda gradient update at t=0 is skipped.
    wealth = 1.0
    wealth_hist = [1.0]
    const = 2 / (2 - np.log(3))
    lambd = 0
    zt2 = 0
    rejected = False
    for t in range(1, len(seq1)):
        St = 1 - lambd * (seq1[t] - seq2[t])
        wealth = wealth * St
        wealth_hist.append(wealth)
        if wealth > 1 / alpha:
            rejected = True          # record but continue for full trajectory
        # Update lambda via ONS
        g = seq1[t] - seq2[t]
        denom = 1 - lambd * g
        z = g / denom if abs(denom) > 1e-12 else g
        zt2 += z ** 2
        lambd = max(min(lambd - const * z / (1 + zt2), 0.5), -0.5)
    # ── end verbatim ──

    return wealth_hist, rejected


def compute_betting_surveillance(
    results_df: pd.DataFrame,
    alpha: float = 0.05,
    min_windows: int = 3,
    bonferroni: bool = True,
) -> pd.DataFrame:
    """Pairwise Chugg et al. betting tests on rolling per-subgroup metrics.

    For each (metric_name, subgroup_config) and each unordered pair of subgroups,
    runs _test_by_betting_ons(seq1, seq2) — testing H0: E[sg1 metric] = E[sg2 metric].

    When *bonferroni* is True (default), alpha is divided by the number of
    pairwise tests within each (metric_name, subgroup_config) group, controlling
    the family-wise error rate (FWER) across subgroup pairs.  For K subgroups
    the correction factor is K*(K-1)/2.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of compute_fairness_over_time or compute_eod_calibration_over_time.
        Required columns: window_center, metric_name, metric_type,
        subgroup_config, subgroup, value.
    alpha : float
        Significance level. Alert threshold = 1/alpha (before Bonferroni).
    min_windows : int
        Minimum number of valid shared windows to run a test (default 3).
    bonferroni : bool
        Apply Bonferroni correction across pairwise tests (default True).

    Returns
    -------
    DataFrame with columns:
        window_center, metric_name, metric_type, subgroup_config,
        pair, wealth, rejected.
    """
    rows = []
    for (mname, mtype, cfg), grp in results_df.groupby(
            ["metric_name", "metric_type", "subgroup_config"]):
        subgroups = sorted(grp["subgroup"].dropna().unique())
        if len(subgroups) < 2:
            continue

        n_pairs = len(subgroups) * (len(subgroups) - 1) // 2
        adj_alpha = alpha / n_pairs if bonferroni and n_pairs > 1 else alpha

        for i, sg1 in enumerate(subgroups):
            for sg2 in subgroups[i + 1:]:
                s1 = (grp[grp["subgroup"] == sg1]
                      .sort_values("window_center")[["window_center", "value"]]
                      .rename(columns={"value": "v1"}))
                s2 = (grp[grp["subgroup"] == sg2]
                      .sort_values("window_center")[["window_center", "value"]]
                      .rename(columns={"value": "v2"}))
                merged = s1.merge(s2, on="window_center")
                ok = merged["v1"].notna() & merged["v2"].notna()
                merged = merged[ok].reset_index(drop=True)
                if len(merged) < min_windows:
                    continue

                pair_label = f"{sg1} vs {sg2}"
                wh, _ = _test_by_betting_ons(
                    merged["v1"].values, merged["v2"].values, alpha=adj_alpha)

                for j, (wc, w) in enumerate(
                        zip(merged["window_center"].values, wh)):
                    rows.append({
                        "window_center": wc,
                        "metric_name": mname, "metric_type": mtype,
                        "subgroup_config": cfg,
                        "pair": pair_label,
                        "wealth": w,
                        "rejected": w > 1 / adj_alpha,
                    })

    return pd.DataFrame(rows)


def plot_betting_surveillance(
    betting_df: pd.DataFrame,
    alpha: float = 0.05,
    figsize: tuple = (6, 4),
) -> Optional[plt.Figure]:
    """Plot Chugg et al. wealth trajectories for pairwise fairness surveillance.

    Parameters
    ----------
    betting_df : pd.DataFrame
        Output of compute_betting_surveillance.
    alpha : float
        Significance level; threshold line at 1/alpha.

    Returns
    -------
    matplotlib Figure, or None if no data.
    """
    if betting_df.empty:
        print("No betting results to plot.")
        return None

    metric_types = [m for m in sorted(betting_df["metric_type"].unique())
                    if "alarm" not in m]
    if not metric_types:
        return None

    threshold = 1.0 / alpha
    n_rows = len(metric_types)
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(figsize[0], figsize[1] * n_rows),
                             sharex=True, squeeze=False)

    for row_idx, mtype in enumerate(metric_types):
        ax = axes[row_idx, 0]
        sub = betting_df[betting_df["metric_type"] == mtype]

        mtype_label = _format_metric_label(mtype)

        pairs = sorted(sub["pair"].unique())
        strat_configs = sorted(sub["subgroup_config"].unique())
        # Color by stratification, linestyle by pair within stratification
        color_map = {cfg: SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)]
                     for i, cfg in enumerate(strat_configs)}
        ls_cycle = ["-", "--", "-."]

        pair_styles: dict = {}
        cfg_pair_count: dict = {}
        for pair in pairs:
            cfg = sub[sub["pair"] == pair]["subgroup_config"].iloc[0]
            n = cfg_pair_count.get(cfg, 0)
            pair_styles[pair] = (color_map[cfg], ls_cycle[n % len(ls_cycle)])
            cfg_pair_count[cfg] = n + 1

        for pair in pairs:
            psub = sub[sub["pair"] == pair].sort_values("window_center")
            c, ls = pair_styles[pair]
            cfg = psub["subgroup_config"].iloc[0]
            cfg_nice = {"sex_label": "Sex", "age_group": "Age"}.get(cfg, cfg)
            label = f"{cfg_nice}: {pair}"
            ax.plot(psub["window_center"].values, psub["wealth"].values,
                    color=c, linestyle=ls, linewidth=2.0, label=label)

        ax.set_yscale("log")
        ax.axhline(threshold, color=PALETTE["red"]["dark"], linewidth=1,
                   linestyle="--", alpha=0.8,
                   label=f"Alert  1/\u03b1 = {threshold:.0f}")
        ax.axhspan(1e-2, threshold, color=PALETTE["green"]["light"], alpha=0.08)
        ax.set_ylabel(f"Wealth\n{mtype_label}", fontsize=8)
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.2)
        if row_idx == 0:
            ax.set_title(
                "Betting-Based Fairness Surveillance  "
                "(Chugg et al. 2023, NeurIPS)",
                fontweight="bold", pad=8)

    axes[-1, 0].set_xlabel("Time")
    fig.tight_layout()
    return fig


# ── Pipeline: rolling fairness computations (ex notebook cells 9-13) ─────────

def compute_fairness_over_time(
    scores_df: pd.DataFrame,
    general_df: pd.DataFrame,
    subgroup_configs: list,
    auc_pairs: Optional[list] = None,
    alarm_columns: Optional[list] = None,
    time_col: str = "datetime",
    patient_col: str = "patient_id",
    general_join_key: str = "patient_id",
    window: Optional[str] = "30D",
    min_events: int = 5,
    min_total: int = 20,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 500,
    ci_alpha: float = 0.05,
    seed: int = 42,
    out_path: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Compute per-subgroup AUROC and/or alarm rate over rolling time windows.

    A single RNG (seeded once) is shared across all windows and subgroups for
    bootstrap CIs.  This creates reproducibility coupling (changing which
    subgroups pass the min_events filter shifts downstream CIs), but does
    not introduce bias: each individual CI remains valid.

    Returns a tidy DataFrame with columns:
      window_center, metric_type, metric_name, subgroup_config, subgroup,
      value, value_lo, value_hi, n_pos, n_total, n_patients
    """
    if not force and out_path is not None and Path(out_path).exists():
        return pd.read_parquet(out_path)

    if auc_pairs is None and alarm_columns is None:
        raise ValueError("Provide at least one of auc_pairs or alarm_columns.")
    auc_pairs     = auc_pairs or []
    alarm_columns = alarm_columns or []

    if window is None:
        window = _infer_window(scores_df[time_col])

    # 1. Merge demographics onto scores
    demo_cols = list({cfg["col"] for cfg in subgroup_configs})
    demo = general_df[[general_join_key] + demo_cols].drop_duplicates(subset=[general_join_key])
    df   = scores_df.merge(demo, left_on=patient_col, right_on=general_join_key,
                           how="left", suffixes=("", "_demo"))
    df[time_col] = pd.to_datetime(df[time_col])

    # 2. Build subgroup columns
    sg_cols = {}
    for cfg in subgroup_configs:
        col = cfg["col"]
        src = col + "_demo" if col + "_demo" in df.columns else col
        sg_col = f"_sg_{cfg['label']}"
        if cfg.get("bins") is not None:
            df[sg_col] = _bin_continuous(df[src], bins=cfg["bins"], labels=cfg.get("bin_labels"))
        elif cfg.get("map") is not None:
            df[sg_col] = df[src].map(cfg["map"])
        else:
            df[sg_col] = df[src].astype(str)
        sg_cols[cfg["label"]] = sg_col

    # 3. Non-overlapping time windows (stride = window, no autocorrelation in CUSUM)
    t_min, t_max = df[time_col].min(), df[time_col].max()
    win_td   = pd.Timedelta(window)
    half_win = win_td / 2
    centers  = pd.date_range(t_min + half_win, t_max - half_win, freq=win_td)
    rng      = np.random.default_rng(seed)

    # 4. Metric definitions
    metrics = []
    for ap in auc_pairs:
        metrics.append(("auc",        ap["score"], ap["label"], f"AUROC \u2013 {ap['name']}"))
    for ac in alarm_columns:
        metrics.append(("alarm_rate", ac,          None,        f"Alarm rate \u2013 {ac}"))

    # 5. Compute
    rows = []
    for center in centers:
        lo_t, hi_t = center - half_win, center + half_win
        w = df[(df[time_col] >= lo_t) & (df[time_col] < hi_t)]
        if w.empty:
            continue

        for mtype, col_a, col_b, mname in metrics:
            for cfg in subgroup_configs:
                sg_col = sg_cols[cfg["label"]]
                for grp_name, grp in w.groupby(sg_col, observed=True):
                    pids = grp[patient_col].values

                    if mtype == "auc":
                        y    = grp[col_b].values
                        s    = grp[col_a].values
                        mask = np.isfinite(y) & np.isfinite(s)
                        y, s, pids_m = y[mask].astype(int), s[mask], pids[mask]
                        n_pos, n_total = int(y.sum()), len(y)
                        if n_pos < min_events or (n_total - n_pos) < min_events or n_total < min_total:
                            rows.append({
                                "window_center": center, "metric_type": mtype,
                                "metric_name": mname, "subgroup_config": cfg["label"],
                                "subgroup": str(grp_name), "value": np.nan,
                                "value_lo": np.nan, "value_hi": np.nan,
                                "n_pos": n_pos, "n_total": n_total,
                                "n_patients": len(np.unique(pids)),
                            })
                            continue
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            val = roc_auc_score(y, s)
                        val_lo = val_hi = np.nan
                        if bootstrap_ci:
                            val_lo, val_hi = _cluster_bootstrap_stat(
                                s, y, pids_m, _auc_fn, n_bootstrap, ci_alpha, rng)

                    elif mtype == "alarm_rate":
                        a    = grp[col_a].values.astype(float)
                        mask = np.isfinite(a)
                        a, pids_m = a[mask], pids[mask]
                        n_total = len(a)
                        n_pos   = int(a.sum())
                        if n_total < min_total:
                            rows.append({
                                "window_center": center, "metric_type": mtype,
                                "metric_name": mname, "subgroup_config": cfg["label"],
                                "subgroup": str(grp_name), "value": np.nan,
                                "value_lo": np.nan, "value_hi": np.nan,
                                "n_pos": 0, "n_total": n_total,
                                "n_patients": len(np.unique(pids)),
                            })
                            continue
                        val = a.mean()
                        val_lo = val_hi = np.nan
                        if bootstrap_ci:
                            val_lo, val_hi = _cluster_bootstrap_stat(
                                a, None, pids_m, _rate_fn, n_bootstrap, ci_alpha, rng)

                    else:
                        continue

                    rows.append({
                        "window_center":  center,
                        "metric_type":    mtype,
                        "metric_name":    mname,
                        "subgroup_config": cfg["label"],
                        "subgroup":       str(grp_name),
                        "value":          val,
                        "value_lo":       val_lo,
                        "value_hi":       val_hi,
                        "n_pos":          n_pos,
                        "n_total":        n_total,
                        "n_patients":     len(np.unique(pids_m)),
                    })

    # Emit NaN rows for subgroups entirely absent from a window
    all_sg = {}
    for cfg in subgroup_configs:
        sg_col = sg_cols[cfg["label"]]
        all_sg[cfg["label"]] = sorted(df[sg_col].dropna().unique().astype(str))

    existing_keys = {
        (r["window_center"], r["metric_name"], r["subgroup_config"], str(r["subgroup"]))
        for r in rows
    }
    for center in centers:
        for mtype_i, col_a, col_b, mname_i in metrics:
            for cfg in subgroup_configs:
                for sg_name in all_sg[cfg["label"]]:
                    if (center, mname_i, cfg["label"], sg_name) not in existing_keys:
                        rows.append({
                            "window_center": center, "metric_type": mtype_i,
                            "metric_name": mname_i, "subgroup_config": cfg["label"],
                            "subgroup": sg_name, "value": np.nan,
                            "value_lo": np.nan, "value_hi": np.nan,
                            "n_pos": 0, "n_total": 0, "n_patients": 0,
                        })

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        print("No subgroup\u2013window combinations met the minimum sample requirements.")
    if out_path is not None:
        results_df.to_parquet(out_path, index=False)
    return results_df


def plot_fairness_over_time(
    results_df: pd.DataFrame,
    bootstrap_ci: bool = True,
    figsize_per_panel: tuple = (14, 4.5),
) -> Optional[plt.Figure]:
    """
    Render a multi-panel time-series plot from the DataFrame returned by
    compute_fairness_over_time(). One panel per (metric_name, subgroup_config) pair.
    """
    if results_df.empty:
        print("results_df is empty \u2014 nothing to plot.")
        return None

    panels = (
        results_df[["metric_name", "metric_type", "subgroup_config"]]
        .drop_duplicates()
        .sort_values(["metric_name", "subgroup_config"])
        .to_dict("records")
    )
    n_panels = len(panels)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * n_panels),
        squeeze=False,
    )

    for panel_idx, panel in enumerate(panels):
        ax    = axes[panel_idx, 0]
        mname = panel["metric_name"]
        mtype = panel["metric_type"]
        cfg_label = panel["subgroup_config"]

        sub = results_df[
            (results_df["metric_name"] == mname)
            & (results_df["subgroup_config"] == cfg_label)
        ]
        for g_idx, grp_name in enumerate(sorted(sub["subgroup"].unique())):
            g = sub[sub["subgroup"] == grp_name].sort_values("window_center")
            c = SUBGROUP_COLORS[g_idx % len(SUBGROUP_COLORS)]
            ax.plot(g["window_center"], g["value"],
                    drawstyle="steps-post", marker="o", ms=4, lw=1.8, color=c, label=grp_name, zorder=3)
            if bootstrap_ci and g["value_lo"].notna().any():
                ax.fill_between(g["window_center"], g["value_lo"], g["value_hi"],
                                alpha=0.15, color=c, zorder=2, step="post")

        if mtype == "auc":
            ax.axhline(0.5, ls="--", lw=0.8, color="grey", alpha=0.5)
            ax.set_ylabel("AUROC")
            ax.set_ylim(0.4, 1.02)
        elif mtype == "tpr_at_fpr":
            ax.set_ylabel("TPR")
            ax.set_ylim(-0.02, 1.02)
        elif mtype == "ece":
            ax.set_ylabel("ECE")
            ax.set_ylim(0.0, None)
        else:
            ax.set_ylabel("Alarm rate")
            ax.set_ylim(0.0, None)

        ax.set_title(f"{mname}  \u2014  stratified by {cfg_label}", fontsize=11)
        ax.legend(
            loc="lower left" if mtype == "auc" else "upper left",
            fontsize=8, framealpha=0.9,
            ncol=min(len(sub["subgroup"].unique()), 4),
        )
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    axes[-1, 0].set_xlabel("Time")
    fig.tight_layout()
    return fig


def compute_eod_calibration_over_time(
    df: pd.DataFrame,
    group_col: str,
    score_label_pairs: list,
    time_col: str = "datetime",
    patient_col: str = "patient_id",
    window: Optional[str] = "30D",
    target_fpr: float = 0.01,
    min_events: int = 5,
    min_total: int = 20,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 200,
    ci_alpha: float = 0.05,
    seed: int = 42,
    out_path: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Compute per-subgroup TPR at fixed FPR and ECE over rolling time windows.

    The global FPR threshold is computed once from the full evaluation dataset
    and applied to every window.  This simulates a deployed operating point
    that is fixed at model-release time (the realistic clinical scenario).
    It is *not* an online/adaptive threshold: early windows are evaluated at a
    threshold that reflects the full study period.  Papers should state that the
    threshold is pre-specified, not learned during monitoring.

    Emits NaN rows for subgroups with insufficient data.
    """
    if not force and out_path is not None and Path(out_path).exists():
        return pd.read_parquet(out_path)

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if window is None:
        window = _infer_window(df[time_col])

    all_groups = sorted(df[group_col].dropna().unique().astype(str))

    # C2: compute global threshold once from the full dataset (stable; per-window
    # threshold is unreliable with only 5–8 negatives in sparse rolling windows).
    global_thresholds = {}
    for pair in score_label_pairs:
        sc_g, lc_g, nm_g = pair["score"], pair["label"], pair["name"]
        y_all = df[lc_g].astype(float).values
        s_all = df[sc_g].astype(float).values
        va = np.isfinite(y_all) & np.isfinite(s_all)
        y_all, s_all = y_all[va].astype(int), s_all[va]
        can_g = y_all.sum() >= min_events and (len(y_all) - y_all.sum()) >= min_events
        global_thresholds[nm_g] = (find_threshold_for_fpr(y_all, s_all, target_fpr)
                                   if can_g else np.nan)

    # C3: non-overlapping windows (stride = window) to avoid autocorrelation in CUSUM.
    t_min, t_max = df[time_col].min(), df[time_col].max()
    win_td = pd.Timedelta(window)
    half_win = win_td / 2
    centers = pd.date_range(t_min + half_win, t_max - half_win, freq=win_td)
    rng = np.random.default_rng(seed)

    rows = []
    for center in centers:
        w = df[(df[time_col] >= center - half_win) & (df[time_col] < center + half_win)]
        if w.empty:
            # Emit NaN for all subgroups and metrics
            for pair in score_label_pairs:
                nm = pair["name"]
                for g_name in all_groups:
                    for mt, mn in [("tpr_at_fpr", f"TPR at {target_fpr:.0%} FPR \u2013 {nm}"),
                                   ("ece", f"ECE \u2013 {nm}")]:
                        rows.append({
                            "window_center": center, "metric_type": mt, "metric_name": mn,
                            "subgroup_config": group_col, "subgroup": g_name,
                            "value": np.nan, "value_lo": np.nan, "value_hi": np.nan,
                            "n_pos": 0, "n_total": 0, "n_patients": 0,
                        })
            continue

        for pair in score_label_pairs:
            sc, lc, nm = pair["score"], pair["label"], pair["name"]
            y_w = w[lc].astype(float).values
            s_w = w[sc].astype(float).values
            valid = np.isfinite(y_w) & np.isfinite(s_w)
            y_w, s_w = y_w[valid].astype(int), s_w[valid]

            # Use pre-computed global threshold (C2)
            thr_w = global_thresholds.get(nm, np.nan)

            w_valid = w.iloc[np.where(valid)[0]]
            seen = set()

            for grp_name, grp in w_valid.groupby(group_col, observed=True):
                seen.add(str(grp_name))
                yg = grp[lc].astype(int).values
                sg = grp[sc].astype(float).values
                pids = grp[patient_col].values
                n_pats = len(np.unique(pids))

                insufficient = (yg.sum() < min_events
                                or (len(yg) - yg.sum()) < min_events
                                or len(yg) < min_total
                                or np.isnan(thr_w))

                # TPR at global threshold
                if insufficient:
                    tpr_val = tpr_lo = tpr_hi = np.nan
                else:
                    tpr_val, fpr_val = compute_tpr_fpr_at_threshold(yg, sg, thr_w)
                    tpr_lo = tpr_hi = np.nan
                    if bootstrap_ci and not np.isnan(tpr_val):
                        def _tpr_boot(sb, lb):
                            return compute_tpr_fpr_at_threshold(lb, sb, thr_w)[0]
                        tpr_lo, tpr_hi = _cluster_bootstrap_stat(
                            sg, yg, pids, _tpr_boot, n_bootstrap, ci_alpha, rng)

                rows.append({
                    "window_center": center, "metric_type": "tpr_at_fpr",
                    "metric_name": f"TPR at {target_fpr:.0%} FPR \u2013 {nm}",
                    "subgroup_config": group_col, "subgroup": str(grp_name),
                    "value": tpr_val, "value_lo": tpr_lo, "value_hi": tpr_hi,
                    "n_pos": int(yg.sum()), "n_total": len(yg), "n_patients": n_pats,
                })

                # ECE — M2: need ≥100 samples for 10 bins × ≥10 obs/bin
                if insufficient or len(yg) < 100:
                    ece_val = ece_lo = ece_hi = np.nan
                else:
                    ece_val = compute_ece(yg, sg)
                    ece_lo = ece_hi = np.nan
                    if bootstrap_ci:
                        def _ece_boot(sb, lb):
                            return compute_ece(lb, sb)
                        ece_lo, ece_hi = _cluster_bootstrap_stat(
                            sg, yg, pids, _ece_boot, n_bootstrap, ci_alpha, rng)

                rows.append({
                    "window_center": center, "metric_type": "ece",
                    "metric_name": f"ECE \u2013 {nm}",
                    "subgroup_config": group_col, "subgroup": str(grp_name),
                    "value": ece_val, "value_lo": ece_lo, "value_hi": ece_hi,
                    "n_pos": int(yg.sum()), "n_total": len(yg), "n_patients": n_pats,
                })

            # Emit NaN for absent subgroups
            for g_name in all_groups:
                if g_name not in seen:
                    for mt, mn in [("tpr_at_fpr", f"TPR at {target_fpr:.0%} FPR \u2013 {nm}"),
                                   ("ece", f"ECE \u2013 {nm}")]:
                        rows.append({
                            "window_center": center, "metric_type": mt, "metric_name": mn,
                            "subgroup_config": group_col, "subgroup": g_name,
                            "value": np.nan, "value_lo": np.nan, "value_hi": np.nan,
                            "n_pos": 0, "n_total": 0, "n_patients": 0,
                        })

    result = pd.DataFrame(rows)
    if out_path is not None:
        result.to_parquet(out_path, index=False)
    return result


def compute_disparity_over_time(
    results_df: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None,
    group_col: Optional[str] = None,
    score_label_pairs: Optional[List[Dict]] = None,
    target_fpr: float = 0.01,
    time_col: str = "datetime",
    patient_col: str = "patient_id",
    window: Optional[str] = "30D",
    bootstrap_ci: bool = False,
    n_bootstrap: int = 200,
    ci_alpha: float = 0.05,
    seed: int = 42,
    out_path: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Compute max-min disparity per window. Optionally with paired bootstrap CIs.

    Disparity is defined as ``max(metric) - min(metric)`` across subgroups
    (range-based).  This choice is scale-preserving (units of the underlying
    metric), avoids undefined ratios when the denominator is zero, and maps
    directly to the CUSUM input for surveillance.
    """
    if not force and out_path is not None and Path(out_path).exists():
        return pd.read_parquet(out_path)

    if window is None:
        centers = sorted(results_df["window_center"].dropna().unique())
        if len(centers) >= 2:
            window = centers[1] - centers[0]
        else:
            window = "30D"

    pair_by_name = {}
    if score_label_pairs:
        pair_by_name = {p["name"]: p for p in score_label_pairs}

    # Compute global thresholds once from raw_df (consistent with point estimates)
    global_thresholds = {}
    if raw_df is not None and score_label_pairs:
        for pair in score_label_pairs:
            sc_g, lc_g, nm_g = pair["score"], pair["label"], pair["name"]
            y_all = raw_df[lc_g].astype(float).values
            s_all = raw_df[sc_g].astype(float).values
            va = np.isfinite(y_all) & np.isfinite(s_all)
            y_all, s_all = y_all[va].astype(int), s_all[va]
            if y_all.sum() >= 5 and (len(y_all) - y_all.sum()) >= 5:
                global_thresholds[nm_g] = find_threshold_for_fpr(y_all, s_all, target_fpr)

    rng = np.random.default_rng(seed)

    rows = []
    for (wc, mtype, mname, cfg_label), grp in results_df.groupby(
            ["window_center", "metric_type", "metric_name", "subgroup_config"]):
        # NB: dropna means the subgroup set contributing to disparity may vary
        # across windows (subgroups with NaN values are excluded).  The n_patients
        # column tracks per-window population for downstream interpretation.
        vals = grp["value"].dropna()

        if len(vals) < 2:
            rows.append({
                "window_center": wc, "metric_type": mtype + "_disparity",
                "metric_name": mname.replace("TPR at ", "EOD at ").replace("ECE", "ECE disparity"),
                "subgroup_config": cfg_label, "subgroup": "disparity",
                "value": np.nan, "value_lo": np.nan, "value_hi": np.nan,
                "n_pos": 0, "n_total": 0, "n_patients": 0,
            })
            continue

        point_disp = vals.max() - vals.min()
        val_lo = val_hi = np.nan

        # Paired bootstrap CI
        if bootstrap_ci and raw_df is not None and group_col is not None:
            # Parse task name from metric_name (after last en-dash)
            task_name = mname.split(" \u2013 ")[-1] if " \u2013 " in mname else mname.split(" -- ")[-1]
            pair = pair_by_name.get(task_name)

            if pair is not None:
                sc, lc = pair["score"], pair["label"]
                win_td = pd.Timedelta(window)
                half_win = win_td / 2
                w = raw_df[
                    (raw_df[time_col] >= wc - half_win) & (raw_df[time_col] < wc + half_win)
                ]

                if len(w) > 0:
                    unique_pats, pat_to_idx = _build_patient_index(
                        w[patient_col].values)

                    boot_disparities = []

                    # Use global threshold for consistency with point estimates
                    g_thr = global_thresholds.get(task_name, np.nan)

                    for _ in range(n_bootstrap):
                        boot_pats = rng.choice(unique_pats, size=len(unique_pats), replace=True)
                        boot_idx = np.concatenate([pat_to_idx[p] for p in boot_pats])
                        w_boot = w.iloc[boot_idx]

                        if mtype == "tpr_at_fpr":
                            boot_thr = g_thr
                            if np.isnan(boot_thr):
                                continue
                        else:
                            boot_thr = np.nan

                        subgroup_vals = []
                        for _, sg in w_boot.groupby(group_col, observed=True):
                            yg = sg[lc].astype(float).values
                            sg_scores = sg[sc].astype(float).values
                            mask = np.isfinite(yg) & np.isfinite(sg_scores)
                            yg, sg_scores = yg[mask].astype(int), sg_scores[mask]
                            v = _compute_subgroup_metric(yg, sg_scores, mtype, boot_thr)
                            if not np.isnan(v):
                                subgroup_vals.append(v)

                        if len(subgroup_vals) >= 2:
                            boot_disparities.append(max(subgroup_vals) - min(subgroup_vals))

                    if len(boot_disparities) > 10:
                        val_lo = np.percentile(boot_disparities, 100 * ci_alpha / 2)
                        val_hi = np.percentile(boot_disparities, 100 * (1 - ci_alpha / 2))

        rows.append({
            "window_center": wc, "metric_type": mtype + "_disparity",
            "metric_name": mname.replace("TPR at ", "EOD at ").replace("ECE", "ECE disparity"),
            "subgroup_config": cfg_label, "subgroup": "disparity",
            "value": point_disp, "value_lo": val_lo, "value_hi": val_hi,
            "n_pos": 0, "n_total": 0, "n_patients": 0,
        })

    result = pd.DataFrame(rows)
    if out_path is not None:
        result.to_parquet(out_path, index=False)
    return result


def plot_disparity_over_time(
    disp_df: pd.DataFrame,
    title_suffix: str = "",
    figsize_per_panel: tuple = (14, 4.5),
) -> Optional[plt.Figure]:
    """Plot disparity over time. Multiple stratifications overlaid on same axes."""
    if disp_df.empty:
        print("No disparity data.")
        return None

    metrics = sorted(disp_df["metric_name"].unique())
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(figsize_per_panel[0], figsize_per_panel[1] * len(metrics)),
                             squeeze=False, sharex=True)

    config_labels = sorted(disp_df["subgroup_config"].unique())
    config_colors = {label: SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)]
                     for i, label in enumerate(config_labels)}

    for i, mname in enumerate(metrics):
        ax = axes[i, 0]
        for cfg_label in config_labels:
            d = disp_df[
                (disp_df["metric_name"] == mname) & (disp_df["subgroup_config"] == cfg_label)
            ].sort_values("window_center")
            if d.empty:
                continue

            c = config_colors[cfg_label]
            ax.plot(d["window_center"], d["value"], marker="o", ms=4, lw=1.8,
                    drawstyle="steps-post", color=c, label=cfg_label, zorder=3)

            if d["value_lo"].notna().any():
                ax.fill_between(d["window_center"], d["value_lo"], d["value_hi"],
                                alpha=0.15, color=c, zorder=2, step="post")

        ax.axhline(0, ls="--", lw=0.6, color="grey", alpha=0.5)
        ax.set_ylabel("Disparity (max \u2212 min)")
        ax.set_title(f"{mname}{title_suffix}", fontsize=10)
        ax.set_ylim(0, None)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    axes[-1, 0].set_xlabel("Time")
    fig.tight_layout()
    return fig


def _plot_composition(results_df, metric_configs, figsize_col=7):
    """Unified composition plot: patient counts + proportions + metric rows.

    Parameters
    ----------
    results_df : DataFrame
        Rolling results with columns: metric_type, metric_name, subgroup_config,
        subgroup, window_center, value, value_lo, value_hi, n_patients.
    metric_configs : list of dict
        Each dict has keys: metric_type, ylabel, ylim, hline (optional).
        Example: [{"metric_type": "auc", "ylabel": "AUROC", "ylim": (0.4, 1.02), "hline": 0.5}]
    """
    if results_df.empty:
        print("Empty results.")
        return None

    # Use first metric_type's rows as reference for composition panels
    ref_type = metric_configs[0]["metric_type"]
    ref_rows = results_df[results_df["metric_type"] == ref_type].copy()
    if ref_rows.empty:
        # Fall back to any available metric
        for mc in metric_configs[1:]:
            ref_rows = results_df[results_df["metric_type"] == mc["metric_type"]].copy()
            if not ref_rows.empty:
                break
    if ref_rows.empty:
        print("No matching metrics found.")
        return None

    cfg_label = ref_rows["subgroup_config"].iloc[0]
    task_names = sorted(ref_rows["metric_name"].unique())
    n_cols = len(task_names)
    n_metric_rows = sum(
        1 for mc in metric_configs
        if not results_df[results_df["metric_type"] == mc["metric_type"]].empty
    )
    if n_metric_rows == 0:
        print("No metrics with data.")
        return None

    fig, axes = plt.subplots(
        2 + n_metric_rows, n_cols,
        figsize=(figsize_col * n_cols, 3 + 4 * n_metric_rows),
        squeeze=False, sharex="col",
        gridspec_kw={"height_ratios": [1, 1] + [2] * n_metric_rows},
    )

    for col_idx, mname in enumerate(task_names):
        sub = ref_rows[
            (ref_rows["subgroup_config"] == cfg_label) & (ref_rows["metric_name"] == mname)
        ]
        subgroups = sorted(sub["subgroup"].unique())
        colors = {sg: SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)] for i, sg in enumerate(subgroups)}
        pivot_n = sub.pivot_table(
            index="window_center", columns="subgroup", values="n_patients", aggfunc="first"
        ).reindex(columns=subgroups)

        # Row 0: patient counts
        ax = axes[0, col_idx]
        for sg in subgroups:
            ax.plot(pivot_n.index, pivot_n[sg], marker="o", ms=3, lw=1.5, color=colors[sg], label=sg)
        ax.set_ylabel("Patients")
        short_name = mname.split(" \u2013 ")[-1] if " \u2013 " in mname else mname
        ax.set_title(f"{short_name}\nstratified by {cfg_label}", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7, ncol=min(len(subgroups), 4))
        ax.grid(axis="y", alpha=0.3)

        # Row 1: proportions
        ax = axes[1, col_idx]
        pf = pivot_n.fillna(0)
        total = pf.sum(axis=1)
        pct = pf.div(total.replace(0, np.nan), axis=0).fillna(0)
        ax.stackplot(pct.index, *[pct[sg] for sg in subgroups],
                     labels=subgroups, colors=[colors[sg] for sg in subgroups], alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion")
        ax.legend(loc="upper right", fontsize=7, ncol=min(len(subgroups), 4))

        # Metric rows
        row_idx = 2
        for mc in metric_configs:
            mt = mc["metric_type"]
            mt_rows = results_df[results_df["metric_type"] == mt]
            if mt_rows.empty:
                continue
            ax = axes[row_idx, col_idx]
            # Match metric_name for this task (handle name mapping like TPR→ECE)
            mc_mname = mc.get("name_transform", lambda m: m)(mname)
            sub_mt = mt_rows[
                (mt_rows["subgroup_config"] == cfg_label) & (mt_rows["metric_name"] == mc_mname)
            ]
            for sg in subgroups:
                g = sub_mt[sub_mt["subgroup"] == sg].sort_values("window_center")
                ax.plot(g["window_center"], g["value"],
                        drawstyle="steps-post", marker="o", ms=3, lw=1.5, color=colors[sg], label=sg)
                if g["value_lo"].notna().any():
                    ax.fill_between(g["window_center"], g["value_lo"], g["value_hi"],
                                    alpha=0.15, color=colors[sg], step="post")
            if mc.get("hline") is not None:
                ax.axhline(mc["hline"], ls="--", color="grey", lw=0.6, alpha=0.5)
            ax.set_ylabel(mc["ylabel"])
            ax.set_ylim(*mc["ylim"])
            ax.grid(axis="y", alpha=0.3)
            row_idx += 1

        # X-axis label on bottom row
        axes[2 + n_metric_rows - 1, col_idx].tick_params(axis="x", rotation=30)
        axes[2 + n_metric_rows - 1, col_idx].set_xlabel("Time")

    fig.tight_layout()
    return fig


def plot_composition_and_auroc(results_df, figsize_col=7):
    """Combined figure: patient counts, proportions, and AUROC over time."""
    return _plot_composition(results_df, [
        {"metric_type": "auc", "ylabel": "AUROC", "ylim": (0.4, 1.02), "hline": 0.5},
    ], figsize_col=figsize_col)


def plot_composition_and_fairness(results_df, figsize_col=7, target_fpr=0.01):
    """Combined figure: patient counts, proportions, TPR@FPR and ECE over time."""
    fpr_pct = f"{target_fpr:.0%}" if target_fpr >= 0.01 else f"{target_fpr:.1%}"
    tpr_label = f"TPR at {fpr_pct} FPR"
    return _plot_composition(results_df, [
        {"metric_type": "tpr_at_fpr", "ylabel": tpr_label, "ylim": (-0.02, 1.02)},
        {"metric_type": "ece", "ylabel": "ECE", "ylim": (0, None),
         "name_transform": lambda m: m.replace(tpr_label, "ECE")},
    ], figsize_col=figsize_col)
