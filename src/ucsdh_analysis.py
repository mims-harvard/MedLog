import itertools
import logging
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from src.config import conf

matplotlib.rcParams["font.family"] = "Arial"

logger = logging.getLogger(__name__)

# Map CSV question text to canonical form (for grouping and display).
# Unmapped questions are used unchanged.
QUESTION_CSV_TO_CANONICAL = {
    "Was the patient enrolled in a clinical for sepsis?": "Was the patient enrolled in a clinical trial for sepsis?",
}


def load_medlog_csv(path: Path) -> pd.DataFrame:
    """Load medlog_data CSV with columns batch, run, csn, question, answer."""
    return pd.read_csv(path)


def _pairwise_agreement(answers: list) -> float:
    """Pairwise agreement among a list of answers: fraction of pairs with same value."""
    if len(answers) < 2:
        return 1.0
    n_agree = sum(1 for a, b in itertools.combinations(answers, 2) if a == b)
    n_pairs = len(answers) * (len(answers) - 1) // 2
    return n_agree / n_pairs if n_pairs else 1.0


def compute_agreement_from_df(
    df: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], list[str], list[str], list[str]]:
    """
    Compute per (csn, question) agreement and return structure for heatmap.

    Returns:
        agreement_dict: data[csn][canonical_question] = agreement
        question_order: ordered list of canonical questions
        question_labels: wrapped display labels for heatmap
        case_order: ordered list of CSNs
    """
    df = df.copy()
    df["question_canonical"] = df["question"].map(lambda q: QUESTION_CSV_TO_CANONICAL.get(q, q))

    # Stable ordering used throughout the heatmap and CSV outputs.
    case_order = sorted(df["csn"].unique())
    question_order = sorted(df["question_canonical"].unique())

    # Agreement definition:
    # 1) group by (csn, question_canonical, run)
    # 2) compute pairwise agreement within each run (using all answers for that run)
    # 3) average those run level agreement scores across runs for each (csn, question_canonical)
    run_agreements = df.groupby(["csn", "question_canonical", "run"], sort=False)["answer"].apply(
        lambda s: _pairwise_agreement(s.astype(str).tolist())
    )

    # Average across runs so the index becomes (csn, question_canonical).
    agreement_by_csn_question = run_agreements.groupby(level=[0, 1]).mean()

    agreement_dict: dict[str, dict[str, float]] = {csn: {} for csn in case_order}
    for (csn, q), val in agreement_by_csn_question.items():
        agreement_dict[str(csn)][str(q)] = float(val)

    question_labels = [textwrap.fill(q, width=40) for q in question_order]
    return agreement_dict, question_order, question_labels, case_order


def save_agreement_matrix_csv(
    agreement_dict: dict[str, dict[str, float]],
    question_order: list[str],
    case_order: list[str],
    path: Path,
) -> None:
    """Save agreement matrix to CSV: rows = questions, columns = CSNs (P1, P2, ...)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    case_labels = [f"P{i + 1}" for i in range(len(case_order))]
    rows = []
    for q in question_order:
        row = {"question": q}
        for j, csn in enumerate(case_order):
            row[case_labels[j]] = agreement_dict.get(csn, {}).get(q, np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    logger.info(f"Saved agreement matrix to {path}")


def save_patient_id_mapping_csv(
    df: pd.DataFrame,
    case_order: list[str],
    path: Path,
) -> None:
    """
    Save CSV mapping heatmap patient labels (P1, P2, ...) to batch and csn.

    P labels are assigned based on the position of each `csn` in `case_order`
    (which should match how `compute_agreement_from_df` defines patient labels).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for i, csn in enumerate(case_order):
        patient_label = f"P{i + 1}"
        csn_batches = sorted(df.loc[df["csn"] == csn, "batch"].astype(str).unique())
        for b in csn_batches:
            rows.append({"patient_label": patient_label, "batch": str(b), "csn": str(csn)})

    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"Saved patient id mapping to {path}")


def warn_if_csn_repeated_across_batches(df: pd.DataFrame) -> None:
    """
    Warn if the same `csn` appears in multiple `batch` values.

    This is a data consistency check; the analysis assumes (for nicer labeling)
    that patients are not duplicated across months or batches.
    """
    if "csn" not in df.columns or "batch" not in df.columns:
        return

    csn_to_n_batches = df.groupby("csn")["batch"].nunique(dropna=True)
    repeated = csn_to_n_batches[csn_to_n_batches > 1].sort_index()
    if repeated.empty:
        return

    n_repeated = len(repeated)
    logger.warning(
        "Found %d patients (CSNs) that appear in more than one batch. This may make P-labels non-unique across batches.",
        n_repeated,
    )

    examples: list[str] = []
    for csn in repeated.index[:10]:
        batches = sorted(df.loc[df["csn"] == csn, "batch"].astype(str).unique().tolist())
        examples.append(f"{csn} -> {', '.join(batches)}")
    logger.warning("Examples (up to 10): %s", "; ".join(examples))


def _build_agreement_matrix_clustered(
    data: dict,
    question_order: list[str] | None,
    question_labels: list[str] | None,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Build agreement matrix and cluster cases. Returns (matrix, case_labels, question_order, question_labels)."""
    cases = list(data.keys())
    if question_order is None:
        all_questions = set()
        for case_data in data.values():
            all_questions.update(case_data.keys())
        question_order = sorted(all_questions)
    if question_labels is None:
        question_labels = [textwrap.fill(q, width=40) for q in question_order]
    questions = question_order
    case_labels = ["P" + str(i + 1) for i in range(len(cases))]

    agreement_matrix = np.zeros((len(questions), len(cases)))
    for i, question in enumerate(questions):
        for j, case in enumerate(cases):
            if question in data[case]:
                agreement_matrix[i, j] = data[case][question]
            else:
                agreement_matrix[i, j] = np.nan

    matrix_for_clustering = agreement_matrix.T
    matrix_for_clustering_clean = np.nan_to_num(matrix_for_clustering, nan=np.nanmean(matrix_for_clustering))
    distances = pdist(matrix_for_clustering_clean, metric="euclidean")
    linkage_matrix = linkage(distances, method="ward")
    cluster_order = leaves_list(linkage_matrix)
    agreement_matrix = agreement_matrix[:, cluster_order]
    case_labels = [case_labels[i] for i in cluster_order]
    return agreement_matrix, case_labels, question_order, question_labels


def create_agreement_heatmap(
    data: dict,
    question_order: list[str] | None = None,
    question_labels: list[str] | None = None,
    figsize=(20, 8),
    annot=True,
    fmt=".2f",
    cbar_label="Average pairwise agreement",
):
    """Create a heatmap of agreement scores from nested dictionary data ``data[case][question]``."""
    agreement_matrix, case_labels, question_order, question_labels = _build_agreement_matrix_clustered(
        data, question_order, question_labels
    )

    fig, ax = plt.subplots(figsize=figsize)

    custom_colors = ["#e76f51", "#f4a261", "#e9c46a", "#32936F"]
    cmap_obj = LinearSegmentedColormap.from_list("custom", custom_colors)

    cmap_obj.set_bad(color="#EBEBEB")

    vmin = np.nanmin(agreement_matrix) if np.any(~np.isnan(agreement_matrix)) else 0.0
    vmax = np.nanmax(agreement_matrix) if np.any(~np.isnan(agreement_matrix)) else 1.0

    annot_array: np.ndarray | bool
    annot_kws = None
    if annot:
        # Only show labels when value != 1.00 (hide perfect agreement cells).
        label_fontsize = 8
        annot_array = np.empty(agreement_matrix.shape, dtype=object)
        for i in range(agreement_matrix.shape[0]):
            for j in range(agreement_matrix.shape[1]):
                v = agreement_matrix[i, j]
                if np.isnan(v) or np.isclose(v, 1.0, atol=1e-12):
                    annot_array[i, j] = ""
                else:
                    annot_array[i, j] = f"{v:{fmt}}"
        annot_kws = {"fontsize": label_fontsize}
    else:
        annot_array = False

    sns.heatmap(
        agreement_matrix,
        xticklabels=case_labels,
        yticklabels=question_labels,
        annot=annot_array,
        fmt="",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": cbar_label},
        ax=ax,
        linewidths=0.5,
        linecolor="#EBEBEB",
        annot_kws=annot_kws,
    )

    ax.set_xlabel("Patient ID", fontsize=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)

    plt.tight_layout()

    return fig, ax


def _batch_color_map(batches: list[str]) -> dict[str, tuple]:
    """Stable mapping from batch to color (Set2 palette) for the batch strip and legend."""
    batches_sorted = sorted(batches)
    colors = ["#92AFD6", "#507DBC", "#375680"]
    return {b: colors[i] for i, b in enumerate(batches_sorted)}


def save_batch_legend_pdf(batches: list[str], path: Path) -> None:
    """Save a standalone PDF legend mapping batch names to colors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = _batch_color_map(batches)
    batches_sorted = sorted(cmap.keys())
    handles = [mpatches.Patch(color=cmap[b], label=b) for b in batches_sorted]
    fig_leg, ax_leg = plt.subplots(figsize=(4, max(1.5, 0.35 * len(batches_sorted))))
    ax_leg.legend(handles=handles, loc="center", frameon=False, fontsize=11)
    ax_leg.axis("off")
    fig_leg.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig_leg)
    logger.info(f"Saved batch legend to {path}")


def create_agreement_heatmap_barplot(
    data: dict,
    df: pd.DataFrame,
    question_order: list[str] | None = None,
    question_labels: list[str] | None = None,
    heatmap_width_ratio: float = 4.0,
    barplot_width_ratio: float = 1.0,
    figsize=(18, 10),
):
    """
    Create a figure with a wide heatmap and a thin barplot.
    Colors in barplot match the heatmap colormap based on the mean value.
    The colorbar is moved to the far right.
    """
    agreement_matrix, case_labels, question_order, question_labels = _build_agreement_matrix_clustered(
        data, question_order, question_labels
    )

    # Compute mean agreement and sort (least to greatest).
    mean_agreement = np.nanmean(agreement_matrix, axis=1)
    sort_indices = np.argsort(mean_agreement)

    agreement_matrix = agreement_matrix[sort_indices, :]
    question_labels = [question_labels[i] for i in sort_indices]
    mean_agreement = mean_agreement[sort_indices]

    n_questions = len(question_order)

    custom_colors = ["#e76f51", "#f4a261", "#e9c46a", "#32936F"]
    cmap_obj = LinearSegmentedColormap.from_list("custom", custom_colors)
    cmap_obj.set_bad(color="#EBEBEB")

    vmin = np.nanmin(agreement_matrix) if np.any(~np.isnan(agreement_matrix)) else 0.0
    vmax = np.nanmax(agreement_matrix) if np.any(~np.isnan(agreement_matrix)) else 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    n_questions = agreement_matrix.shape[0]
    n_patients = agreement_matrix.shape[1]

    # Record counts per patient for the top strip.
    # `case_labels` are "P1..Pk" assigned from the sorted csn list before clustering,
    # then reordered by clustering. Recover csn for each patient label.
    case_order = list(data.keys())  # original insertion order == compute_agreement_from_df's sorted csn order
    df_csn = df["csn"].astype(str)
    counts_by_csn = df.assign(_csn_str=df_csn).groupby("_csn_str").size()

    patient_csn_in_display_order: list[str] = []
    record_counts_in_display_order: list[int] = []
    patient_batch_per_column: list[str] = []
    for plabel in case_labels:
        idx = int(plabel[1:]) - 1
        csn = str(case_order[idx])
        patient_csn_in_display_order.append(csn)
        record_counts_in_display_order.append(int(counts_by_csn.get(csn, 0)))
        csn_batches = sorted(df.loc[df["csn"].astype(str) == csn, "batch"].astype(str).unique().tolist())
        if len(csn_batches) > 1:
            logger.warning(
                "CSN %s appears in multiple batches (%s). Using %s for batch annotation strip.",
                csn,
                ", ".join(csn_batches),
                csn_batches[0],
            )
        patient_batch_per_column.append(csn_batches[0] if csn_batches else "")

    batches_for_legend = sorted(df["batch"].astype(str).unique().tolist())
    batch_to_color = _batch_color_map(batches_for_legend)
    batch_bar_colors = [batch_to_color.get(b, (0.7, 0.7, 0.7)) for b in patient_batch_per_column]

    # Layout: left column [medlog strip, batch strip, heatmap];
    # right column = barplot aligned to heatmap row only.
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[0.12, 0.03, 1.0],  # make batch strip much thinner
        width_ratios=[heatmap_width_ratio, barplot_width_ratio],
        wspace=0.05,
        hspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_batch = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_heat = fig.add_subplot(gs[2, 0], sharex=ax_top)
    ax_bar = fig.add_subplot(gs[2, 1], sharey=ax_heat)

    x_positions = np.arange(n_patients) + 0.5

    # MedLog record count strip (top row)
    ax_top.bar(
        x_positions,
        record_counts_in_display_order,
        color="#6c757d",
        edgecolor="black",
        linewidth=0.4,
        width=0.9,
        alpha=0.9,
    )
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_top.set_ylabel("")
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    ax_top.spines["left"].set_visible(True)
    ax_top.spines["left"].set_color("black")
    ax_top.spines["left"].set_linewidth(1)

    # Batch annotation strip (middle row, much thinner)
    ax_batch.bar(
        x_positions,
        np.ones(n_patients),
        color=batch_bar_colors,
        edgecolor="black",
        linewidth=0.4,
        width=0.9,
        alpha=0.95,
    )
    ax_batch.set_ylim(0, 1.0)
    ax_batch.set_yticks([])
    ax_batch.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_batch.spines.values():
        spine.set_visible(False)

    # No colorbar here; a separate one is added aligned to the heatmap below.
    sns.heatmap(
        agreement_matrix,
        xticklabels=case_labels,
        yticklabels=question_labels,
        annot=False,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        ax=ax_heat,
        linewidths=0.5,
        linecolor="#EBEBEB",
    )
    ax_heat.set_xlabel("Patient ID", fontsize=12, labelpad=10)

    # Keep top strips aligned with heatmap columns.
    ax_batch.set_xlim(ax_heat.get_xlim())
    ax_top.set_xlim(ax_heat.get_xlim())

    # Show full spines around the heatmap only.
    for spine in ax_heat.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)

    # Map mean_agreement onto the same scale as the heatmap so bar colors match.
    bar_colors = [cmap_obj(norm(np.nan_to_num(v, nan=vmin))) for v in mean_agreement]

    x_values = [x + 0.5 for x in range(n_questions)]
    ax_bar.barh(
        x_values,
        mean_agreement,
        color=bar_colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_bar.set_xlabel("Mean pairwise agreement", fontsize=12)
    ax_bar.set_xlim(vmin, vmax)

    # Keep the y order explicit and consistent across shared axes:
    # top = least agreement, bottom = greatest agreement.
    ax_heat.set_ylim(n_questions, 0)
    ax_bar.set_ylim(n_questions, 0)

    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Add colorbar on the far right with reserved space (avoid overlap with ax_bar)
    heatmap_pos = ax_heat.get_position()
    cbar_ax = fig.add_axes([0.94, heatmap_pos.y0, 0.015, heatmap_pos.height])
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Pairwise agreement", fontsize=12)

    # Reserve space for the colorbar and legend.
    plt.subplots_adjust(left=0.2, right=0.92, wspace=0.05, hspace=0.05)

    return fig


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics: n_patients (LLM used), avg_queries_per_patient.
    Overall row plus one row per batch. Queries = number of question and answer rows per patient.
    """
    overall = pd.DataFrame([
        {
            "batch": "overall",
            "n_patients": df["csn"].nunique(),
            "avg_queries_per_patient": df.groupby("csn").size().mean(),
        }
    ])
    # avg_queries_per_patient per batch = mean of (rows per patient) within that batch
    by_batch = (
        df.groupby("batch", group_keys=False)
        .apply(
            lambda g: pd.Series({
                "n_patients": g["csn"].nunique(),
                "avg_queries_per_patient": g.groupby("csn").size().mean(),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    return pd.concat([overall, by_batch], ignore_index=True)


def main(  # noqa: C901
    medlog_path: Path | None = None,
    output_path: Path | None = None,
    matrix_path: Path | None = None,
    summary_path: Path | None = None,
    batch: str | None = None,
) -> Path:
    """
    Load medlog CSV, compute agreement, save matrix CSV, generate heatmap, save summary stats.

    Args:
        medlog_path: Path to medlog_data.csv. Defaults to config.
        output_path: Path to save heatmap PDF. Defaults to config.
        matrix_path: Path to save agreement matrix CSV. Defaults to config.
        summary_path: Path to save summary statistics CSV. Defaults to config.
        batch: If set, filter data to this batch and save outputs under figures/<batch>/<batch>_XXX.

    Returns:
        Path to the saved heatmap file.
    """
    if medlog_path is None:
        medlog_path = conf.ucsdh.medlog_data_path
    if batch is not None:
        batch_dir = conf.ucsdh.figures_dir / batch
        batch_dir.mkdir(parents=True, exist_ok=True)
        output_path = batch_dir / f"{batch}_agreement_heatmap.pdf"
        matrix_path = batch_dir / f"{batch}_agreement_matrix.csv"
        summary_path = batch_dir / f"{batch}_agreement_summary.csv"
        barplot_path = batch_dir / f"{batch}_agreement_heatmap_barplot.pdf"
    else:
        if output_path is None:
            output_path = conf.ucsdh.heatmap_path
        if matrix_path is None:
            matrix_path = conf.ucsdh.agreement_matrix_path
        if summary_path is None:
            summary_path = conf.ucsdh.summary_stats_path
        barplot_path = conf.ucsdh.heatmap_barplot_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading medlog data from {medlog_path}")
    if not medlog_path.exists():
        msg = f"Medlog data file not found: {medlog_path}"
        raise FileNotFoundError(msg)
    df = load_medlog_csv(medlog_path)

    warn_if_csn_repeated_across_batches(df)

    if batch is not None:
        df = df[df["batch"] == batch].copy()
        logger.info(f"Filtered to batch {batch}: {len(df)} rows, {df['csn'].nunique()} patients")
        if len(df) == 0:
            msg = f"No rows found for batch {batch!r}"
            raise ValueError(msg)

    agreement_dict, question_order, question_labels, case_order = compute_agreement_from_df(df)
    logger.info(f"Computed agreement for {len(case_order)} cases, {len(question_order)} questions")

    if batch is not None:
        patient_id_path = conf.ucsdh.figures_dir / batch / "patient_id.csv"
    else:
        patient_id_path = conf.ucsdh.figures_dir / "patient_id.csv"
    save_patient_id_mapping_csv(df, case_order, patient_id_path)

    save_agreement_matrix_csv(agreement_dict, question_order, case_order, matrix_path)

    logger.info("Creating agreement heatmap")
    create_agreement_heatmap(
        agreement_dict,
        question_order=question_order,
        question_labels=question_labels,
    )

    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    logger.info(f"Saved heatmap to {output_path}")
    svg_path = output_path.with_suffix(".svg")
    plt.savefig(svg_path, bbox_inches="tight", format="svg")
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, bbox_inches="tight", dpi=600, format="png")
    plt.close()

    logger.info("Creating agreement heatmap with barplot")
    fig_bar = create_agreement_heatmap_barplot(
        agreement_dict,
        df=df,
        question_order=question_order,
        question_labels=question_labels,
    )
    fig_bar.savefig(barplot_path, bbox_inches="tight", format="pdf")
    logger.info(f"Saved agreement heatmap barplot to {barplot_path}")
    fig_bar.savefig(barplot_path.with_suffix(".svg"), bbox_inches="tight", format="svg")
    fig_bar.savefig(barplot_path.with_suffix(".png"), bbox_inches="tight", dpi=600, format="png")
    plt.close(fig_bar)

    if batch is not None:
        batch_legend_path = conf.ucsdh.figures_dir / batch / "batch_legend.pdf"
    else:
        batch_legend_path = conf.ucsdh.figures_dir / "batch_legend.pdf"
    save_batch_legend_pdf(sorted(df["batch"].astype(str).unique().tolist()), batch_legend_path)

    summary_df = compute_summary_stats(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary statistics to {summary_path}")

    return output_path
