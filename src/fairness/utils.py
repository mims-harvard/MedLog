"""Fairness core metrics, constants, and helper utilities."""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Constants ──────────────────────────────────────────────────────────────────

PALETTE = {
    "red":    {"dark": "#E64B35", "light": "#F39B7F"},
    "blue":   {"light": "#4DBBD5", "dark": "#8491B4"},
    "green":  {"dark": "#00A087", "light": "#91D1C2"},
    "brown":  {"dark": "#7E6148", "light": "#B09C85"},
    "purple": {"dark": "#6A3D9A", "light": "#CAB2D6"},
    "yellow": {"dark": "#D4B013", "light": "#F3E08C"},
    "accent_blue": "#3C5488",
    "accent_red":  "#DC0000",
}
SEX_PALETTE = {"Male": PALETTE["accent_blue"], "Female": PALETTE["red"]["dark"]}
SUBGROUP_COLORS = [
    PALETTE["accent_blue"], PALETTE["red"]["dark"],  PALETTE["green"]["dark"],
    PALETTE["purple"]["dark"], PALETTE["yellow"]["dark"], PALETTE["brown"]["dark"],
    PALETTE["blue"]["light"],  PALETTE["red"]["light"],
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _strip_prefix(metric_name):
    """'AUROC – Circulatory failure (8 h)' -> 'Circulatory failure (8 h)'"""
    for sep in [" \u2013 ", " -- "]:
        if sep in metric_name:
            return metric_name.split(sep, 1)[1]
    return metric_name


def _build_patient_index(patient_ids):
    """Map each unique patient to an array of row indices.

    Returns (unique_patients, pat_to_idx) where pat_to_idx maps patient →
    np.ndarray of integer indices into the original array.
    """
    unique = np.unique(patient_ids)
    idx_map = {p: [] for p in unique}
    for i, p in enumerate(patient_ids):
        idx_map[p].append(i)
    return unique, {p: np.array(v) for p, v in idx_map.items()}


# ── Core metrics ──────────────────────────────────────────────────────────────

def cluster_bootstrap_auc(y_true, y_score, patient_ids,
                           n_bootstrap=1000, ci_alpha=0.05, seed=42,
                           return_samples=False):
    if y_true.sum() < 1 or (len(y_true) - y_true.sum()) < 1:
        if return_samples:
            return np.nan, np.nan, np.nan, []
        return np.nan, np.nan, np.nan
    point = roc_auc_score(y_true, y_score)
    unique_pats, pat_to_idx = _build_patient_index(patient_ids)

    rng = np.random.default_rng(seed)
    boot_aucs = []
    for _ in range(n_bootstrap):
        boot_pats = rng.choice(unique_pats, size=len(unique_pats), replace=True)
        unique_boot_pats, counts = np.unique(boot_pats, return_counts=True)
        idx = np.concatenate([np.tile(pat_to_idx[p], c) for p, c in zip(unique_boot_pats, counts)])

        yb, sb = y_true[idx], y_score[idx]
        # Skip degenerate resamples where AUC is undefined (all labels
        # identical).  Alternative: assign AUC=0.5, but that assumes a
        # random classifier, biasing the distribution in the opposite
        # direction.  Rejection sampling is standard practice.
        if yb.sum() < 1 or (len(yb) - yb.sum()) < 1:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot_aucs.append(roc_auc_score(yb, sb))

    if len(boot_aucs) < max(10, n_bootstrap // 10):
        warnings.warn(
            f"Only {len(boot_aucs)}/{n_bootstrap} bootstrap iterations produced valid AUC; "
            "CI may be unreliable",
            stacklevel=2,
        )
        if len(boot_aucs) == 0:
            if return_samples:
                return point, np.nan, np.nan, []
            return point, np.nan, np.nan
    lo = np.percentile(boot_aucs, 100 * ci_alpha / 2)
    hi = np.percentile(boot_aucs, 100 * (1 - ci_alpha / 2))
    if return_samples:
        return point, lo, hi, boot_aucs
    return point, lo, hi


def _bin_continuous(series, bins, labels=None):
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def _cluster_bootstrap_stat(values, labels, patient_ids, stat_fn, n_bootstrap, ci_alpha, rng):
    unique_pats, pat_to_idx = _build_patient_index(patient_ids)

    boot_stats = []
    n_failed = 0
    for _ in range(n_bootstrap):
        boot_pats = rng.choice(unique_pats, size=len(unique_pats), replace=True)
        unique_boot_pats, counts = np.unique(boot_pats, return_counts=True)
        boot_idx = np.concatenate([np.tile(pat_to_idx[p], c) for p, c in zip(unique_boot_pats, counts)])

        vb = values[boot_idx]
        lb = labels[boot_idx] if labels is not None else None
        try:
            boot_stats.append(stat_fn(vb, lb))
        except Exception:
            n_failed += 1
            continue

    if n_failed > n_bootstrap // 2:
        warnings.warn(
            f"_cluster_bootstrap_stat: {n_failed}/{n_bootstrap} bootstrap replicates "
            f"failed — stat_fn may be misconfigured",
            stacklevel=2,
        )

    min_valid = max(10, n_bootstrap // 10)
    if len(boot_stats) >= min_valid:
        return (np.percentile(boot_stats, 100 * ci_alpha / 2),
                np.percentile(boot_stats, 100 * (1 - ci_alpha / 2)))
    return np.nan, np.nan


def _auc_fn(scores, labels):
    """Bootstrap stat function for AUROC. Called as stat_fn(values, labels)."""
    if labels.sum() < 1 or (len(labels) - labels.sum()) < 1:
        raise ValueError
    return roc_auc_score(labels, scores)


def _rate_fn(scores, _labels):
    """Bootstrap stat function for alarm/event rate."""
    return scores.mean()


def _compute_subgroup_metric(y_true, y_score, metric_type, global_threshold=np.nan):
    """Compute a single metric value for one subgroup (used by paired bootstrap)."""
    if len(y_true) == 0:
        return np.nan
    if metric_type == "tpr_at_fpr":
        if y_true.sum() < 1:
            return np.nan
        return compute_tpr_fpr_at_threshold(y_true, y_score, global_threshold)[0]
    elif metric_type == "auc":
        if y_true.sum() < 1 or (len(y_true) - y_true.sum()) < 1:
            return np.nan
        return roc_auc_score(y_true, y_score)
    elif metric_type == "ece":
        if len(y_true) < 10:
            return np.nan
        return compute_ece(y_true, y_score)
    elif metric_type == "alarm_rate":
        return y_score.mean()
    return np.nan


def find_threshold_for_fpr(y_true, y_score, target_fpr=0.01):
    """Find the decision threshold achieving *target_fpr* on the full population.

    Uses ``method='higher'`` so the threshold is always at or above the exact
    quantile, guaranteeing achieved FPR <= *target_fpr* even with tied scores.
    Assumes continuous model outputs (neural-network probabilities); with heavily
    discretised scores the achieved FPR may be noticeably below the target.
    """
    neg_scores = y_score[y_true == 0]
    if len(neg_scores) == 0:
        return np.nan
    return np.quantile(neg_scores, 1.0 - target_fpr, method='higher')


def compute_tpr_fpr_at_threshold(y_true, y_score, threshold):
    """Compute TPR and FPR at a fixed decision threshold."""
    y_pred = (y_score >= threshold).astype(int)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = y_pred[pos_mask].mean() if pos_mask.sum() > 0 else np.nan
    fpr = y_pred[neg_mask].mean() if neg_mask.sum() > 0 else np.nan
    return tpr, fpr


def _uniform_bin_masks(y_prob, n_bins=10):
    """Return (bin_edges, masks) for uniform [0,1] probability bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    masks = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == 0:
            masks.append((y_prob >= lo) & (y_prob <= hi))
        else:
            masks.append((y_prob > lo) & (y_prob <= hi))
    return edges, masks


def compute_ece(y_true, y_prob, n_bins=10, strategy="uniform"):
    """Expected Calibration Error.

    Parameters
    ----------
    strategy : 'uniform' (default) or 'quantile'
        'uniform'  — equal-width bins on [0, 1].
        'quantile' — equal-count bins; recommended for skewed score distributions
        (e.g. clinical alarm scores with <5 % event prevalence) where uniform
        binning concentrates nearly all samples in the first bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return np.nan

    if strategy == "quantile":
        if len(y_true) < n_bins * 5:
            n_bins = max(1, len(y_true) // 5)
        sort_idx = np.argsort(y_prob)
        y_s = y_true[sort_idx]
        p_s = y_prob[sort_idx]
        bins = np.array_split(np.arange(len(y_s)), n_bins)
        ece = 0.0
        n = len(y_true)
        for b in bins:
            if len(b) == 0:
                continue
            bin_acc = y_s[b].mean()
            bin_conf = p_s[b].mean()
            ece += (len(b) / n) * abs(bin_acc - bin_conf)
        return ece

    # uniform
    _, masks = _uniform_bin_masks(y_prob, n_bins)
    ece = 0.0
    n_binned = sum(m.sum() for m in masks)
    if n_binned == 0:
        return np.nan
    for mask in masks:
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n_binned) * abs(bin_acc - bin_conf)
    return ece


def compute_reliability_curve(y_true, y_prob, n_bins=15, strategy='quantile'):
    """Return (midpoints, frac_pos, counts, se) for reliability diagrams.

    Parameters
    ----------
    strategy : 'quantile' (default) or 'uniform'
        'quantile' — equal-count bins; avoids empty bins in sparse high-probability regions.
        'uniform'  — equal-width probability bins (original behaviour).
    se : binomial standard error sqrt(p*(1-p)/n) per bin; NaN for empty bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == 'quantile':
        if len(y_true) < n_bins * 5:
            n_bins_orig = n_bins
            n_bins = max(1, len(y_true) // 5)
            if n_bins < n_bins_orig:
                warnings.warn(
                    f"compute_reliability_curve: n_bins reduced {n_bins_orig}→{n_bins} "
                    f"(need ≥5 obs/bin, got {len(y_true)} samples)",
                    stacklevel=2,
                )
        sort_idx = np.argsort(y_prob)
        y_s = y_true[sort_idx]
        p_s = y_prob[sort_idx]
        bins = np.array_split(np.arange(len(y_s)), n_bins)
        midpoints = np.array([p_s[b].mean() if len(b) else np.nan for b in bins])
        frac_pos  = np.array([y_s[b].mean()  if len(b) else np.nan for b in bins])
        counts    = np.array([len(b)                                 for b in bins], dtype=int)
    else:  # uniform
        bin_edges, masks = _uniform_bin_masks(y_prob, n_bins)
        midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        frac_pos  = np.full(n_bins, np.nan)
        counts    = np.zeros(n_bins, dtype=int)
        for b, mask in enumerate(masks):
            counts[b] = int(mask.sum())
            if mask.sum() > 0:
                frac_pos[b] = y_true[mask].mean()

    se = np.where(
        counts > 0,
        np.sqrt(frac_pos * (1.0 - frac_pos) / np.maximum(counts, 1)),
        np.nan,
    )
    return midpoints, frac_pos, counts, se


def _extract_ys_pids(grp, score_col, label_col, patient_col="patient_id"):
    """Extract finite (y, scores, patient_ids) from a DataFrame group.

    Labels are rounded to {0, 1} via ``np.round``.  This is a defensive cast
    for binary endpoint labels that may have floating-point representation noise
    after parquet serialisation (e.g. 0.9999999 or 1.0000001).  The function
    assumes upstream labels are binary; feeding soft / probabilistic labels will
    silently binarise them at the 0.5 boundary.
    """
    y = grp[label_col].astype(float).values
    s = grp[score_col].astype(float).values
    pids = grp[patient_col].values
    mask = np.isfinite(y) & np.isfinite(s)
    return np.round(y[mask]).astype(int), s[mask], pids[mask]
