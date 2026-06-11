"""Fairness analysis package — metrics, visualizations, and surveillance."""

# Core metrics and constants
from .utils import (  # noqa: F401
    PALETTE, SEX_PALETTE, SUBGROUP_COLORS,
    cluster_bootstrap_auc,
    find_threshold_for_fpr, compute_tpr_fpr_at_threshold,
    compute_ece, compute_reliability_curve,
)

# Static visualizations and computations
from .static import (  # noqa: F401
    plot_subgroup_auroc,
    plot_grouped_fairness_bars,
    plot_equalized_odds_scatter, plot_fairness_radar,
    compute_static_auroc, compute_static_eod, compute_static_calibration,
    compute_static_disparity, compute_results_all_df,
)

# Dynamic / rolling analysis + surveillance
from .dynamic import (  # noqa: F401
    reshape_rolling_to_static, plot_disparity_trajectory, plot_radar_evolution,
    compute_cusum, plot_disparity_stacked, plot_cusum_surveillance,
    plot_disparity_and_cusum,
    compute_betting_surveillance, plot_betting_surveillance,
    compute_fairness_over_time, plot_fairness_over_time,
    compute_eod_calibration_over_time, compute_disparity_over_time,
    plot_disparity_over_time, plot_composition_and_auroc, plot_composition_and_fairness,
)

# Test data generation
from .testdata import make_dummy_assembled  # noqa: F401
