"""Static fairness analysis: computations and visualizations for cross-sectional metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import (
    PALETTE, SEX_PALETTE, SUBGROUP_COLORS, _strip_prefix,
    cluster_bootstrap_auc, find_threshold_for_fpr,
    compute_tpr_fpr_at_threshold, compute_ece,
    _cluster_bootstrap_stat, _extract_ys_pids,
)

_DEFAULT_SUBGROUP_ORDER = ["Overall", "Male", "Female", "<50", "50\u201365", "65+"]


def plot_subgroup_auroc(results_all_df, score_label_pairs, figsize=(5, 5), ylim=None,
                        out_dir=None, out_prefix="", out_formats=("png", "pdf"), dpi=300):
    """Per-task subgroup AUROC bar chart, one figure per task.

    Left panel: AUROC with 95% cluster-bootstrap CI (vertical bars).
    Right panel: population composition as vertical stacked bars (Sex, Age).

    Parameters
    ----------
    ylim : tuple (ymin, ymax) or None
        Fixed y-axis limits for the AUROC panel. If None, auto-scaled from data.
    out_dir : str or None
        If set, save each figure to this directory.
    out_prefix : str
        Filename prefix for saved figures (e.g. ``"v5_perf_bars"``).

    Returns a list of (fig, task_slug) tuples.
    """
    if score_label_pairs is None:
        raise ValueError("score_label_pairs is required")

    subgroup_order = _DEFAULT_SUBGROUP_ORDER
    n_sg = len(subgroup_order)
    x = np.arange(n_sg)
    figs = []

    for pair in score_label_pairs:
        task_name = pair["name"]
        task_slug = "circ" if "circ" in pair["score"] else "resp"

        fig, (ax_perf, ax_n) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={"width_ratios": [2.5, 1], "wspace": 0.35},
        )

        all_vals = []
        for sg_idx, sg in enumerate(subgroup_order):
            c = SUBGROUP_COLORS[sg_idx % len(SUBGROUP_COLORS)]
            row = results_all_df[
                (results_all_df["task"] == task_name) & (results_all_df["subgroup"] == sg)
            ]
            if row.empty or np.isnan(row.iloc[0]["auroc"]):
                continue
            r = row.iloc[0]
            val = r["auroc"]
            err_lo = max(0.0, val - r["auroc_lo"])
            err_hi = max(0.0, r["auroc_hi"] - val)
            all_vals.append(val)
            ax_perf.bar(
                sg_idx, val, 0.65,
                yerr=[[err_lo], [err_hi]],
                color=c, alpha=0.55, edgecolor="white", linewidth=0.5,
                error_kw={"linewidth": 0.8, "capsize": 3, "capthick": 0.8},
                zorder=3,
            )

        if ylim is not None:
            ax_perf.set_ylim(*ylim)
        elif all_vals:
            y_min = max(0.0, np.floor(min(all_vals) * 20) / 20 - 0.02)
            y_max = min(1.0, max(all_vals) + 0.04)
            ax_perf.set_ylim(y_min, y_max)
        ax_perf.set_xticks(x)
        ax_perf.set_xticklabels(subgroup_order, fontsize=7, rotation=30, ha="right")
        ax_perf.set_ylabel("AUROC")
        fig.suptitle(task_name, fontweight="bold", fontsize=9, y=0.98)
        ax_perf.spines["top"].set_visible(False)
        ax_perf.spines["right"].set_visible(False)

        # ── Right: population composition (two vertical stacked bars) ────────
        _strat_groups = {
            "Sex": [sg for sg in subgroup_order if sg in ("Male", "Female")],
            "Age": [sg for sg in subgroup_order if sg in ("<50", "50\u201365", "65+")],
        }

        # Collect n_patients per subgroup (skip Overall)
        _sg_n = {}
        for sg_idx, sg in enumerate(subgroup_order):
            if sg == "Overall":
                continue
            row = results_all_df[
                (results_all_df["task"] == task_name) & (results_all_df["subgroup"] == sg)
            ]
            n_raw = row.iloc[0]["n_patients"] if not row.empty else 0
            _sg_n[sg] = int(n_raw) if not (row.empty or np.isnan(n_raw)) else 0

        x_positions = list(range(len(_strat_groups)))
        for x_pos, (strat_name, sg_list) in enumerate(_strat_groups.items()):
            total = sum(_sg_n.get(sg, 0) for sg in sg_list)
            bottom = 0.0
            for sg in sg_list:
                sg_idx = subgroup_order.index(sg)
                n_pat = _sg_n.get(sg, 0)
                c = SUBGROUP_COLORS[sg_idx % len(SUBGROUP_COLORS)]
                frac = n_pat / total if total > 0 else 0
                ax_n.bar(x_pos, frac, bottom=bottom, color=c, alpha=0.55,
                         edgecolor="white", linewidth=0.8, width=0.55)
                if frac > 0.08:
                    ax_n.text(x_pos, bottom + frac / 2, f"{sg}\n{n_pat}",
                              ha="center", va="center", fontsize=5.5)
                bottom += frac

        total_all = sum(_sg_n.get(sg, 0) for sg in _strat_groups["Sex"])
        ax_n.set_ylim(0, 1)
        ax_n.set_xticks(x_positions)
        ax_n.set_xticklabels(list(_strat_groups.keys()), fontsize=7)
        ax_n.set_ylabel(f"Proportion (N={total_all})", fontsize=7)
        ax_n.yaxis.set_tick_params(labelsize=6)
        ax_n.spines["top"].set_visible(False)
        ax_n.spines["right"].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figs.append((fig, task_slug))

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pfx = f"{out_prefix}_" if out_prefix else ""
        for fig_i, slug_i in figs:
            for fmt in out_formats:
                fig_i.savefig(Path(out_dir) / f"{pfx}{slug_i}.{fmt}",
                              dpi=dpi, bbox_inches="tight")

    return figs


def plot_grouped_fairness_bars(results_all_df, score_label_pairs=None, figsize=(14, 5)):
    """Two-panel grouped bar chart: (a) AUROC by subgroup, (b) AUROC disparity."""
    if score_label_pairs is None:
        raise ValueError("score_label_pairs is required")
    tasks = [p["name"] for p in score_label_pairs]
    n_tasks = len(tasks)
    task_colors = SUBGROUP_COLORS[:max(n_tasks, 2)]

    subgroup_order = _DEFAULT_SUBGROUP_ORDER

    fig, (ax_perf, ax_disp) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [3.5, 1], "wspace": 0.35},
    )

    # Panel (a): Subgroup AUROC
    x = np.arange(len(subgroup_order))
    bar_width = 0.32
    all_vals = []
    for t_idx, task_name in enumerate(tasks):
        vals, errs_lo, errs_hi = [], [], []
        for sg in subgroup_order:
            row = results_all_df[
                (results_all_df["task"] == task_name)
                & (results_all_df["subgroup"] == sg)
            ]
            if row.empty or np.isnan(row.iloc[0]["auroc"]):
                vals.append(np.nan); errs_lo.append(0); errs_hi.append(0)
            else:
                r = row.iloc[0]
                vals.append(r["auroc"])
                errs_lo.append(max(0, r["auroc"] - r["auroc_lo"]))
                errs_hi.append(max(0, r["auroc_hi"] - r["auroc"]))
                all_vals.append(r["auroc"])

        offset = (t_idx - (n_tasks - 1) / 2) * bar_width
        bars = ax_perf.bar(
            x + offset, vals, bar_width,
            yerr=[errs_lo, errs_hi],
            color=task_colors[t_idx], alpha=0.55,
            edgecolor="white", linewidth=0.5,
            error_kw={"linewidth": 0.8, "capsize": 3, "capthick": 0.8},
            label=task_name, zorder=3,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax_perf.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=5.5, fontweight="bold", rotation=45,
                )

    ax_perf.axvline(0.5, color="grey", linewidth=0.4, linestyle=":", alpha=0.5)
    ax_perf.axvline(2.5, color="grey", linewidth=0.4, linestyle=":", alpha=0.5)

    if all_vals:
        y_min = max(0, np.floor(min(all_vals) * 20) / 20 - 0.02)
        y_max = min(1.0, max(all_vals) + 0.03)
    else:
        y_min, y_max = 0, 1
    ax_perf.set_ylim(y_min, y_max)

    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(subgroup_order, fontsize=8)
    ax_perf.set_ylabel("AUROC")
    ax_perf.legend(loc="lower left", fontsize=7, framealpha=0.9)
    ax_perf.set_title("Subgroup performance", fontweight="bold", pad=8)
    ax_perf.text(-0.06, 1.06, "a", transform=ax_perf.transAxes,
                 fontsize=12, fontweight="bold", va="top")

    # Panel (b): Disparity
    strat_map = {"Sex": "sex_label", "Age": "age_group"}
    x_disp = np.arange(len(strat_map))
    for t_idx, task_name in enumerate(tasks):
        disp_vals = []
        for gv in strat_map.values():
            sub = results_all_df[
                (results_all_df["task"] == task_name)
                & (results_all_df["group_var"] == gv)
            ]
            valid = sub["auroc"].dropna()
            disp_vals.append(valid.max() - valid.min() if len(valid) >= 2 else np.nan)

        offset = (t_idx - (n_tasks - 1) / 2) * bar_width
        bars = ax_disp.bar(
            x_disp + offset, disp_vals, bar_width,
            color=task_colors[t_idx], alpha=0.55,
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        for bar, v in zip(bars, disp_vals):
            if not np.isnan(v):
                ax_disp.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0003,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6, fontweight="bold",
                )

    ax_disp.set_xticks(x_disp)
    ax_disp.set_xticklabels(list(strat_map.keys()), fontsize=8)
    ax_disp.set_ylabel("AUROC disparity (max \u2212 min)")
    ax_disp.set_title("Disparity", fontweight="bold", pad=8)
    ax_disp.text(-0.18, 1.06, "b", transform=ax_disp.transAxes,
                 fontsize=12, fontweight="bold", va="top")

    fig.tight_layout()
    return fig


def plot_equalized_odds_scatter(eod_results_df, score_label_pairs=None,
                                target_fpr=0.01,
                                group_vars=("sex_label", "age_group"),
                                figsize_per_panel=(6, 5.5)):
    """Scatter plot of (FPR, TPR) per subgroup at a global threshold."""
    if score_label_pairs is None:
        raise ValueError("score_label_pairs is required")
    tasks = [p["name"] for p in score_label_pairs]
    n_tasks = len(tasks)
    marker_map = {"sex_label": "o", "age_group": "^", "age_sex": "s"}
    gv_labels = {"sex_label": "Sex", "age_group": "Age", "age_sex": "Age \u00d7 Sex"}

    # Build a unified color map for subgroups across all group_vars
    sg_color_idx = 0
    sg_colors = {}
    for gv in group_vars:
        sub = eod_results_df[eod_results_df["group_var"] == gv]
        for sg in sorted(sub["subgroup"].unique()):
            if sg not in sg_colors:
                sg_colors[sg] = SUBGROUP_COLORS[sg_color_idx % len(SUBGROUP_COLORS)]
                sg_color_idx += 1

    fig, axes = plt.subplots(
        1, n_tasks,
        figsize=(figsize_per_panel[0] * n_tasks, figsize_per_panel[1]),
        squeeze=False,
    )

    for t_idx, task_name in enumerate(tasks):
        ax = axes[0, t_idx]
        all_tpr, all_fpr = [], []
        legend_seen = set()
        tpr_by_gv = {gv: [] for gv in group_vars}

        for gv in group_vars:
            sub = eod_results_df[
                (eod_results_df["task"] == task_name)
                & (eod_results_df["group_var"] == gv)
            ].copy()
            sub = sub.dropna(subset=["tpr", "fpr"])
            if sub.empty:
                continue

            marker = marker_map.get(gv, "o")
            for _, row in sub.iterrows():
                sg_name = row["subgroup"]
                color = sg_colors.get(sg_name, "grey")
                yerr_lo = max(0, row["tpr"] - row["tpr_lo"]) if not np.isnan(row["tpr_lo"]) else 0
                yerr_hi = max(0, row["tpr_hi"] - row["tpr"]) if not np.isnan(row["tpr_hi"]) else 0
                label_str = f'{sg_name} ({gv_labels.get(gv, gv)})'
                ax.errorbar(
                    row["fpr"], row["tpr"],
                    yerr=[[yerr_lo], [yerr_hi]],
                    fmt=marker, color=color, markersize=9,
                    markeredgecolor="white", markeredgewidth=0.8,
                    elinewidth=1.2, capsize=3, capthick=0.8,
                    label=label_str if label_str not in legend_seen else "_nolegend_",
                    zorder=5,
                )
                legend_seen.add(label_str)
                all_tpr.append(row["tpr"])
                all_fpr.append(row["fpr"])
                tpr_by_gv[gv].append(row["tpr"])

        # target FPR reference line
        ax.axvline(target_fpr, color="grey", linewidth=0.8, linestyle="--",
                   alpha=0.6, label=f"Target FPR = {target_fpr}")

        # Per-group_var disparity annotations
        ann_y = 0.03
        for gv in group_vars:
            tprs = tpr_by_gv[gv]
            if len(tprs) >= 2:
                delta = max(tprs) - min(tprs)
                ax.annotate(
                    f"\u0394 TPR ({gv_labels[gv]}) = {delta:.3f}",
                    xy=(0.97, ann_y), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9),
                )
                ann_y += 0.06

        # Smart axis limits
        if all_fpr and all_tpr:
            fpr_pad = max(0.005, (max(all_fpr) - min(all_fpr)) * 0.5)
            tpr_pad = max(0.03, (max(all_tpr) - min(all_tpr)) * 0.5)
            ax.set_xlim(max(0, min(all_fpr) - fpr_pad),
                        max(all_fpr) + fpr_pad)
            ax.set_ylim(max(0, min(all_tpr) - tpr_pad),
                        min(1.02, max(all_tpr) + tpr_pad))

        # Text labels
        for gv in group_vars:
            sub = eod_results_df[
                (eod_results_df["task"] == task_name)
                & (eod_results_df["group_var"] == gv)
            ].dropna(subset=["tpr", "fpr"])
            for _, row in sub.iterrows():
                color = sg_colors.get(row["subgroup"], "grey")
                ax.annotate(
                    row["subgroup"],
                    (row["fpr"], row["tpr"]),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=6.5, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color,
                                    lw=0.4, alpha=0.5),
                )

        ax.set_xlabel("False positive rate (FPR)")
        ax.set_ylabel("True positive rate (TPR)")
        ax.set_title(task_name, fontweight="bold", pad=12)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left", fontsize=6,
                  framealpha=0.9, ncol=1)
        ax.text(-0.1, 1.08, chr(ord("a") + t_idx), transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

    fig.tight_layout()
    return fig


def plot_fairness_radar(results_all_df, eod_results_df, cal_results_df,
                        score_label_pairs=None, figsize=(13, 6)):
    """Radar chart showing multi-metric fairness fingerprint per subgroup."""
    if score_label_pairs is None:
        raise ValueError("score_label_pairs is required")
    tasks = [p["name"] for p in score_label_pairs]
    short_names = {t: t.replace(" failure", "").replace(" (8 h)", "") for t in tasks}

    metric_labels = []
    for t in tasks:
        sn = short_names[t]
        metric_labels.extend([
            f"AUROC\n{sn}",
            f"TPR@1%FPR\n{sn}",
            f"Calibration\n{sn}",
        ])
    n_metrics = len(metric_labels)

    def _get_values(group_var, subgroup):
        vals = []
        for t in tasks:
            r = results_all_df[
                (results_all_df["task"] == t)
                & (results_all_df["group_var"] == group_var)
                & (results_all_df["subgroup"] == subgroup)
            ]
            vals.append(r.iloc[0]["auroc"] if not r.empty and not np.isnan(r.iloc[0]["auroc"]) else np.nan)
            r = eod_results_df[
                (eod_results_df["task"] == t)
                & (eod_results_df["group_var"] == group_var)
                & (eod_results_df["subgroup"] == subgroup)
            ]
            vals.append(r.iloc[0]["tpr"] if not r.empty and not np.isnan(r.iloc[0]["tpr"]) else np.nan)
            r = cal_results_df[
                (cal_results_df["task"] == t)
                & (cal_results_df["group_var"] == group_var)
                & (cal_results_df["subgroup"] == subgroup)
            ]
            ece = r.iloc[0]["ece"] if not r.empty and not np.isnan(r.iloc[0]["ece"]) else np.nan
            vals.append(1.0 - ece if not np.isnan(ece) else np.nan)
        return vals

    panels = [
        ("sex_label", ["Male", "Female"], "Sex"),
        ("age_group", ["<50", "50\u201365", "65+"], "Age"),
    ]

    fig = plt.figure(figsize=figsize)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    all_vals = []
    for gv, subgroups, _ in panels:
        for sg in subgroups:
            all_vals.extend([v for v in _get_values(gv, sg) if not np.isnan(v)])

    if not all_vals:
        return fig

    val_min = min(all_vals)
    val_max = max(all_vals)
    margin = max(0.02, (val_max - val_min) * 0.3)
    r_min = max(0, round(val_min - margin, 2))
    r_max = min(1.0, round(val_max + margin / 2, 2))
    if r_max - r_min < 0.05:
        r_min = max(0, r_max - 0.1)

    panel_colors = {
        "sex_label": [PALETTE["accent_blue"], PALETTE["red"]["dark"]],
        "age_group": [PALETTE["green"]["dark"], PALETTE["purple"]["dark"], PALETTE["yellow"]["dark"]],
    }

    for ax_idx, (gv, subgroups, title) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), ax_idx + 1, polar=True)
        colors = panel_colors.get(gv, SUBGROUP_COLORS[:len(subgroups)])

        for sg_idx, sg in enumerate(subgroups):
            vals = _get_values(gv, sg)
            color = colors[sg_idx % len(colors)]

            plot_vals = [v if not np.isnan(v) else np.nan for v in vals]
            plot_vals_closed = plot_vals + plot_vals[:1]

            ax.plot(angles_closed, plot_vals_closed, color=color,
                    linewidth=1.8, label=sg, zorder=3)
            # Fill only non-NaN segments
            fill_vals = [v if not np.isnan(v) else r_min for v in vals]
            fill_closed = fill_vals + fill_vals[:1]
            if any(not np.isnan(v) for v in vals):
                ax.fill(angles_closed, fill_closed, color=color, alpha=0.10)
            for a, v in zip(angles, vals):
                if not np.isnan(v):
                    ax.plot(a, v, "o", color=color, markersize=4.5, zorder=4)

        ax.set_ylim(r_min, r_max)
        ax.set_thetagrids(
            [a * 180 / np.pi for a in angles],
            metric_labels, fontsize=7.5,
        )
        n_rings = 4
        ring_vals = np.linspace(r_min, r_max, n_rings + 1)[1:]
        ax.set_rgrids(
            ring_vals,
            [f"{v:.2f}" for v in ring_vals],
            angle=45, fontsize=6, color="grey", alpha=0.7,
        )
        ax.set_title(title, fontweight="bold", pad=25, fontsize=11)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
                  fontsize=8, framealpha=0.9)
        ax.text(-0.15, 1.15, chr(ord("a") + ax_idx), transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

    fig.subplots_adjust(wspace=0.45)
    return fig


# ── Static computation functions (ex notebook cells 22-33) ───────────────────

def compute_static_auroc(assembled, score_label_pairs, group_var,
                         n_bootstrap=1000, ci_alpha=0.05, seed=42,
                         min_events=5, verbose=False, return_samples=False):
    """Compute AUROC per subgroup for a single stratification variable.

    Returns a DataFrame with columns:
      task, subgroup, n_patients, n_pos, prevalence, auroc, auroc_lo, auroc_hi
      (+ boot_samples: list of bootstrap AUROCs when return_samples=True)
    """
    rows = []
    for pair in score_label_pairs:
        score_col, label_col, name = pair["score"], pair["label"], pair["name"]
        for grp_name, grp in assembled.groupby(group_var, observed=True):
            y, s, pids = _extract_ys_pids(grp, score_col, label_col)
            n_pos, n_patients = int(y.sum()), len(np.unique(pids))

            base = {"task": name, "subgroup": str(grp_name),
                    "n_patients": n_patients, "n_pos": n_pos,
                    "prevalence": n_pos / len(y) if len(y) > 0 else np.nan}

            if n_pos < min_events or (len(y) - n_pos) < min_events:
                rows.append({**base,
                             "auroc": np.nan, "auroc_lo": np.nan, "auroc_hi": np.nan,
                             **({"boot_samples": []} if return_samples else {})})
                if verbose:
                    print(f"  [{name} / {grp_name}] too few events ({n_pos}) \u2014 AUROC=N/A")
                continue

            result = cluster_bootstrap_auc(y, s, pids, n_bootstrap, ci_alpha, seed,
                                           return_samples=return_samples)
            if return_samples:
                auc, lo, hi, samples = result
                rows.append({**base, "auroc": auc, "auroc_lo": lo, "auroc_hi": hi,
                             "boot_samples": samples})
            else:
                auc, lo, hi = result
                rows.append({**base, "auroc": auc, "auroc_lo": lo, "auroc_hi": hi})
            if verbose:
                print(f"  [{name} / {grp_name}]  AUROC={auc:.3f} [{lo:.3f}\u2013{hi:.3f}]  "
                      f"n={n_patients} patients, {n_pos} events")

    return pd.DataFrame(rows)


def compute_static_eod(assembled, score_label_pairs, group_vars=None,
                       target_fpr=0.01, n_bootstrap=1000, ci_alpha=0.05,
                       seed=42, min_events=5, verbose=False):
    """Compute TPR and FPR at a fixed global threshold per subgroup.

    The threshold is set at the population level to achieve *target_fpr* and
    then applied identically to every subgroup.  Per-subgroup FPR will
    generally differ from the target.  This is intentional: equalized-odds
    analysis at a fixed operating point compares TPR across subgroups at the
    *same* decision boundary, simulating a deployed model whose threshold is
    pre-specified at release time.

    Returns a DataFrame with columns:
      task, group_var, subgroup, tpr, tpr_lo, tpr_hi, fpr,
      global_threshold, n_patients, n_pos
    """
    if group_vars is None:
        group_vars = ["sex_label", "age_group", "age_sex"]

    rows = []
    for pair in score_label_pairs:
        score_col, label_col, name = pair["score"], pair["label"], pair["name"]

        y_all = assembled[label_col].astype(float).values
        s_all = assembled[score_col].astype(float).values
        valid = np.isfinite(y_all) & np.isfinite(s_all)
        y_all, s_all = y_all[valid].astype(int), s_all[valid]

        global_thr = find_threshold_for_fpr(y_all, s_all, target_fpr)
        if verbose:
            tpr_ov, fpr_ov = compute_tpr_fpr_at_threshold(y_all, s_all, global_thr)
            print(f"\n{name}: global threshold={global_thr:.4f}  overall TPR={tpr_ov:.3f}  FPR={fpr_ov:.4f}")

        for group_col in group_vars:
            if group_col not in assembled.columns:
                continue
            for grp_name, grp in assembled.groupby(group_col, observed=True):
                y, s, pids = _extract_ys_pids(grp, score_col, label_col)

                if y.sum() < min_events or (len(y) - y.sum()) < min_events:
                    rows.append({"task": name, "group_var": group_col, "subgroup": str(grp_name),
                                 "tpr": np.nan, "tpr_lo": np.nan, "tpr_hi": np.nan,
                                 "fpr": np.nan, "global_threshold": global_thr,
                                 "n_patients": len(np.unique(pids)), "n_pos": int(y.sum())})
                    continue

                tpr, fpr = compute_tpr_fpr_at_threshold(y, s, global_thr)

                def _tpr_fn(scores_b, labels_b):
                    return compute_tpr_fpr_at_threshold(labels_b, scores_b, global_thr)[0]
                sub_rng = np.random.default_rng(seed)
                tpr_lo, tpr_hi = _cluster_bootstrap_stat(
                    s, y, pids, _tpr_fn, n_bootstrap, ci_alpha, sub_rng)

                rows.append({"task": name, "group_var": group_col, "subgroup": str(grp_name),
                             "tpr": tpr, "tpr_lo": tpr_lo, "tpr_hi": tpr_hi,
                             "fpr": fpr, "global_threshold": global_thr,
                             "n_patients": len(np.unique(pids)), "n_pos": int(y.sum())})

    return pd.DataFrame(rows)


def compute_static_calibration(assembled, score_label_pairs, group_vars=None,
                               n_bootstrap=1000, ci_alpha=0.05, seed=42,
                               min_samples=30):
    """Compute ECE per subgroup with bootstrap CIs.

    Returns a DataFrame with columns:
      task, group_var, subgroup, ece, ece_lo, ece_hi, n_patients
    """
    if group_vars is None:
        group_vars = ["sex_label", "age_group", "age_sex"]

    def _ece_fn(scores_b, labels_b):
        return compute_ece(labels_b, scores_b)

    rows = []
    for pair in score_label_pairs:
        score_col, label_col, name = pair["score"], pair["label"], pair["name"]
        for group_col in group_vars:
            if group_col not in assembled.columns:
                continue
            for grp_name, grp in assembled.groupby(group_col, observed=True):
                y, s, pids = _extract_ys_pids(grp, score_col, label_col)

                if len(y) < min_samples:
                    rows.append({"task": name, "group_var": group_col, "subgroup": str(grp_name),
                                 "ece": np.nan, "ece_lo": np.nan, "ece_hi": np.nan,
                                 "n_patients": len(np.unique(pids))})
                    continue

                ece_val = compute_ece(y, s)
                sub_rng = np.random.default_rng(seed)
                ece_lo, ece_hi = _cluster_bootstrap_stat(
                    s, y, pids, _ece_fn, n_bootstrap, ci_alpha, sub_rng)
                rows.append({"task": name, "group_var": group_col, "subgroup": str(grp_name),
                             "ece": ece_val, "ece_lo": ece_lo, "ece_hi": ece_hi,
                             "n_patients": len(np.unique(pids))})

    return pd.DataFrame(rows)


def compute_static_disparity(eod_results_df, cal_results_df=None):
    """Compute disparity summary from EOD and calibration results.

    Disparity is the range (max - min) across subgroups.  This is
    scale-preserving, avoids undefined ratios when the denominator is zero,
    and is interpretable in the units of the metric (probability).

    Returns a DataFrame with columns:
      Task, Stratification, EOD (TPR gap), Best group, Worst group, ECE disparity
    """
    rows = []
    for (task, gv), sub_eod in eod_results_df.groupby(["task", "group_var"]):
        valid_tpr = sub_eod["tpr"].dropna()
        if len(valid_tpr) < 2:
            eod_val = np.nan
            best_grp = worst_grp = "N/A"
        else:
            eod_val = valid_tpr.max() - valid_tpr.min()
            best_grp = sub_eod.loc[sub_eod["tpr"].idxmax(), "subgroup"]
            worst_grp = sub_eod.loc[sub_eod["tpr"].idxmin(), "subgroup"]

        ece_disp = np.nan
        if cal_results_df is not None:
            sub_cal = cal_results_df[
                (cal_results_df["task"] == task) & (cal_results_df["group_var"] == gv)
            ]
            valid_ece = sub_cal["ece"].dropna() if not sub_cal.empty else pd.Series(dtype=float)
            ece_disp = valid_ece.max() - valid_ece.min() if len(valid_ece) >= 2 else np.nan

        rows.append({
            "Task": task, "Stratification": gv,
            "EOD (TPR gap)": eod_val,
            "Best group": best_grp, "Worst group": worst_grp,
            "ECE disparity": ece_disp,
        })

    return pd.DataFrame(rows)


def compute_results_all_df(assembled, score_label_pairs, results_by_sex_df=None,
                           n_bootstrap=1000, ci_alpha=0.05, seed=42,
                           min_events=5, out_path=None, force=False):
    """Compute unified AUROC DataFrame: overall + sex + age_group.

    If *out_path* is given and the file already exists, load and return it
    instead of recomputing.

    Returns a DataFrame with columns:
      task, group_var, subgroup, n_patients, n_pos, prevalence,
      auroc, auroc_lo, auroc_hi
    """
    if not force and out_path is not None and Path(out_path).exists():
        return pd.read_parquet(out_path)

    kw = dict(n_bootstrap=n_bootstrap, ci_alpha=ci_alpha, seed=seed, min_events=min_events)
    parts = []

    # Overall: use a synthetic constant column
    assembled_ov = assembled.copy()
    assembled_ov["_overall"] = "Overall"
    overall = compute_static_auroc(assembled_ov, score_label_pairs, "_overall", **kw)
    overall["group_var"] = "overall"
    parts.append(overall)

    # Per sex: reuse precomputed results or compute fresh
    if results_by_sex_df is not None:
        sex_df = results_by_sex_df.copy()
        sex_df["group_var"] = "sex_label"
        parts.append(sex_df[overall.columns])
    else:
        sex_df = compute_static_auroc(assembled, score_label_pairs, "sex_label", **kw)
        sex_df["group_var"] = "sex_label"
        parts.append(sex_df)

    # Per age group
    age_df = compute_static_auroc(assembled, score_label_pairs, "age_group", **kw)
    age_df["group_var"] = "age_group"
    parts.append(age_df)

    result = pd.concat(parts, ignore_index=True)
    if out_path is not None:
        result.to_parquet(out_path, index=False)
    return result
