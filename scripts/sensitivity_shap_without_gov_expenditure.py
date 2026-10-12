"""SHAP sensitivity analysis excluding government expenditure.

Reviewer #3 raised the possibility that attribution rankings are sensitive to
the high correlation between ``GDP_Total`` and ``Gov_Expenditure``.  This
standalone script compares the manuscript's full-feature interpretation model
with an otherwise identical model that excludes only ``Gov_Expenditure``.

Both models use the main panel, the same full-sample Winsorization, the fixed
XGBoost interpretation-model parameters, and exact TreeSHAP values used by the
main SHAP figures.  No existing analysis script is changed.

Outputs
-------
results/SHAP_Without_Gov_Expenditure_Rank_Comparison.csv
results/SHAP_Without_Gov_Expenditure_Summary.csv
results/FigureS_SHAP_Importance_Without_Gov_Expenditure.pdf / .png
results/FigureS_GDP_Dependence_Comparison.pdf / .png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr

from src.config import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    GROUP_COL,
    TO_100M_RMB,
    XGB_FINAL_PARAMS,
    apply_journal_style,
    save_figure,
    write_table,
)
from src.explain import analysis_sample, fit_full_model, shap_matrix
from src.preprocessing import load_panel


DROPPED_FEATURE = "Gov_Expenditure"
FOCAL_FEATURE = "GDP_Total"


def mean_absolute_shap(values: np.ndarray, columns: pd.Index) -> pd.Series:
    """Return descending mean absolute SHAP importance with feature labels."""
    return pd.Series(np.abs(values).mean(axis=0), index=columns).sort_values(
        ascending=False
    )


def rank_comparison(
    baseline_importance: pd.Series,
    reduced_importance: pd.Series,
) -> pd.DataFrame:
    """Combine importance values and ranks, retaining the excluded feature."""
    baseline_rank = baseline_importance.rank(
        ascending=False, method="min"
    ).astype(int)
    reduced_rank = reduced_importance.rank(ascending=False, method="min").astype(int)

    rows = []
    for feature in baseline_importance.index:
        if feature in reduced_importance.index:
            reduced_value = float(reduced_importance[feature])
            reduced_feature_rank = int(reduced_rank[feature])
            rank_change = reduced_feature_rank - int(baseline_rank[feature])
            shap_change = reduced_value - float(baseline_importance[feature])
        else:
            reduced_value = np.nan
            reduced_feature_rank = np.nan
            rank_change = np.nan
            shap_change = np.nan

        rows.append(
            {
                "Feature": feature,
                "Baseline_Mean_Absolute_SHAP": float(baseline_importance[feature]),
                "Baseline_Rank": int(baseline_rank[feature]),
                "Reduced_Mean_Absolute_SHAP": reduced_value,
                "Reduced_Rank": reduced_feature_rank,
                "Rank_Change_Reduced_Minus_Baseline": rank_change,
                "Mean_Absolute_SHAP_Change": shap_change,
            }
        )

    return pd.DataFrame(rows)


def plot_reduced_importance(values: np.ndarray, X: pd.DataFrame) -> None:
    """Create the reduced-model counterpart to the main SHAP bar chart."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        values,
        X,
        plot_type="bar",
        show=False,
        color=COLOR_SECONDARY,
        plot_size=(10, 8),
    )
    plt.title(
        "SHAP Importance Without Government Expenditure",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=12)
    plt.tight_layout()
    figure = plt.gcf()
    save_figure(figure, "FigureS_SHAP_Importance_Without_Gov_Expenditure")
    plt.close(figure)


def plot_gdp_dependence_comparison(
    X_baseline: pd.DataFrame,
    baseline_values: np.ndarray,
    X_reduced: pd.DataFrame,
    reduced_values: np.ndarray,
    baseline_rank: int,
    reduced_rank: int,
) -> None:
    """Compare GDP dependence under the full and reduced specifications."""
    panels = [
        (
            X_baseline,
            baseline_values,
            COLOR_PRIMARY,
            f"Baseline (GDP rank: {baseline_rank})",
        ),
        (
            X_reduced,
            reduced_values,
            COLOR_SECONDARY,
            f"Without Gov_Expenditure (GDP rank: {reduced_rank})",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, (X, values, color, title) in zip(axes, panels):
        x_values = X[FOCAL_FEATURE].to_numpy() / TO_100M_RMB
        gdp_shap = values[:, X.columns.get_loc(FOCAL_FEATURE)]
        ax.scatter(x_values, gdp_shap, alpha=0.6, color=color, s=15)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Total GDP Actual Value (100 Million Yuan)", fontsize=11)
        ax.set_ylabel("SHAP Value (Impact on CO2)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "GDP_Total SHAP Dependence: Baseline vs. Reduced Model",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(pad=3.0)
    save_figure(fig, "FigureS_GDP_Dependence_Comparison")
    plt.close(fig)


def main() -> None:
    apply_journal_style()

    panel = load_panel()
    X_baseline, y = analysis_sample(panel)

    missing_required = [
        feature
        for feature in (DROPPED_FEATURE, FOCAL_FEATURE)
        if feature not in X_baseline.columns
    ]
    if missing_required:
        raise KeyError(f"Required feature(s) not found: {missing_required}")

    # The main SHAP preprocessing is applied once. Because Winsorization is
    # column-wise, removing one column afterward preserves every common-feature
    # value exactly while changing only the feature set supplied to XGBoost.
    X_reduced = X_baseline.drop(columns=[DROPPED_FEATURE])
    if not X_baseline.drop(columns=[DROPPED_FEATURE]).equals(X_reduced):
        raise RuntimeError("Reduced predictors differ by more than the target feature")

    baseline_model = fit_full_model(X_baseline, y)
    reduced_model = fit_full_model(X_reduced, y)
    baseline_values = shap_matrix(baseline_model, X_baseline)
    reduced_values = shap_matrix(reduced_model, X_reduced)

    baseline_importance = mean_absolute_shap(baseline_values, X_baseline.columns)
    reduced_importance = mean_absolute_shap(reduced_values, X_reduced.columns)
    comparison = rank_comparison(baseline_importance, reduced_importance)
    write_table(comparison, "SHAP_Without_Gov_Expenditure_Rank_Comparison")

    baseline_ranks = baseline_importance.rank(
        ascending=False, method="min"
    ).astype(int)
    reduced_ranks = reduced_importance.rank(ascending=False, method="min").astype(int)
    common_features = X_reduced.columns.tolist()
    spearman_correlation, spearman_p_value = spearmanr(
        baseline_ranks.loc[common_features],
        reduced_ranks.loc[common_features],
    )

    gdp_baseline_rank = int(baseline_ranks[FOCAL_FEATURE])
    gdp_reduced_rank = int(reduced_ranks[FOCAL_FEATURE])
    gdp_baseline_shap = float(baseline_importance[FOCAL_FEATURE])
    gdp_reduced_shap = float(reduced_importance[FOCAL_FEATURE])
    gdp_shap_change = gdp_reduced_shap - gdp_baseline_shap
    gdp_shap_percent_change = 100.0 * gdp_shap_change / gdp_baseline_shap

    summary = pd.DataFrame(
        [
            {
                "Observations": len(panel),
                "Cities": panel[GROUP_COL].nunique(),
                "Baseline_Features": X_baseline.shape[1],
                "Reduced_Features": X_reduced.shape[1],
                "Removed_Feature": DROPPED_FEATURE,
                "Spearman_Rank_Correlation": float(spearman_correlation),
                "Spearman_P_Value": float(spearman_p_value),
                "GDP_Total_Baseline_Rank": gdp_baseline_rank,
                "GDP_Total_Reduced_Rank": gdp_reduced_rank,
                "GDP_Total_Rank_Change_Reduced_Minus_Baseline": (
                    gdp_reduced_rank - gdp_baseline_rank
                ),
                "GDP_Total_Baseline_Mean_Absolute_SHAP": gdp_baseline_shap,
                "GDP_Total_Reduced_Mean_Absolute_SHAP": gdp_reduced_shap,
                "GDP_Total_Mean_Absolute_SHAP_Change": gdp_shap_change,
                "GDP_Total_Mean_Absolute_SHAP_Percent_Change": (
                    gdp_shap_percent_change
                ),
            }
        ]
    )
    write_table(summary, "SHAP_Without_Gov_Expenditure_Summary")

    plot_reduced_importance(reduced_values, X_reduced)
    plot_gdp_dependence_comparison(
        X_baseline,
        baseline_values,
        X_reduced,
        reduced_values,
        gdp_baseline_rank,
        gdp_reduced_rank,
    )

    print(f"[model] XGBoost parameters: {XGB_FINAL_PARAMS}")
    print(f"[sample] {len(panel)} observations; {panel[GROUP_COL].nunique()} cities")
    print("\nBaseline mean absolute SHAP ranking:")
    print(baseline_importance.rename("Mean |SHAP|").to_string())
    print("\nReduced mean absolute SHAP ranking:")
    print(reduced_importance.rename("Mean |SHAP|").to_string())
    print(f"\n[rank stability] Spearman rho = {spearman_correlation:.6f}")
    print(
        f"[GDP_Total] rank {gdp_baseline_rank} -> {gdp_reduced_rank}; "
        f"mean |SHAP| {gdp_baseline_shap:.6f} -> {gdp_reduced_shap:.6f} "
        f"({gdp_shap_percent_change:+.2f}%)"
    )
    print("[saved] reduced SHAP importance and GDP dependence comparison figures")


if __name__ == "__main__":
    main()
