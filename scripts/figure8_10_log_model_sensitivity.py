"""Figures 8 to 10 - Sensitivity analysis on a scale-adjusted outcome.

The main analysis models total emissions, which may partly reflect city size.
This script repeats the framework with the natural logarithm of emissions as
the outcome and with the public-bus count removed, since that variable is
strongly collinear with the economic aggregates reported in Figure 1. Both
changes are aimed at the same concern: whether the identified patterns
survive once absolute urban scale is taken out of the model.

The outcome is log-transformed here, so the metrics below are in log units and
are not directly comparable with Tables 2 and 3.

Produced here:
  Figure 8  - feature importance ranking of the log-outcome model
  Figure 9  - SHAP dependence of Total GDP under the log-outcome model
  Figure 10 - SHAP dependence of the industrial electricity ratio

Outputs: results/Figure8_LogModel_Feature_Importance.pdf / .png
         results/Figure9_LogModel_SHAP_Dependence_GDP.pdf / .png
         results/Figure10_LogModel_SHAP_Dependence_IndElectricity.pdf / .png
         results/Sensitivity_LogModel_CV.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold

from src.config import (
    COLOR_PRIMARY,
    GROUP_COL,
    RANDOM_STATE,
    TARGET,
    TO_100M_RMB,
    apply_journal_style,
    save_figure,
    write_table,
)
from src.explain import fit_full_model, prepare_frame, shap_matrix
from src.models import PARAM_GRIDS, benchmark_models
from src.preprocessing import feature_matrix, load_panel

N_SPLITS = 5
N_INNER_SPLITS = 3


def nested_r2(X: pd.DataFrame, y: pd.Series, groups: pd.Series, pipeline, grid, outer_cv, grouped: bool):
    """Unbiased R-squared for the log-outcome model under nested CV."""
    scores = []
    # KFold ignores the groups argument, so only pass it under GroupKFold.
    splits = outer_cv.split(X, y, groups) if grouped else outer_cv.split(X, y)
    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv = (
            GroupKFold(n_splits=N_INNER_SPLITS)
            if grouped
            else KFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        )
        search = GridSearchCV(pipeline, grid, cv=inner_cv, scoring="r2", n_jobs=-1)
        if grouped:
            search.fit(X_train, y_train, groups=groups.iloc[train_idx])
        else:
            search.fit(X_train, y_train)
        scores.append(r2_score(y_test, search.best_estimator_.predict(X_test)))

    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    apply_journal_style()

    panel = load_panel()
    # The nested cross-validation starts from raw features: the pipeline
    # applies its own winsorization inside each training fold.
    X_raw = feature_matrix(panel, drop_buses=True)
    X = prepare_frame(panel, drop_buses=True)
    y_log = np.log(panel[TARGET].astype(float))
    groups = panel[GROUP_COL]

    xgb_pipeline = benchmark_models()["XGBoost"]
    grid = PARAM_GRIDS["XGBoost"]

    print("--- Nested cross-validation, log-transformed outcome ---")
    kf_mean, kf_std = nested_r2(X_raw, y_log, groups, xgb_pipeline, grid,
                                KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE), False)
    print(f"Standard KFold R2 : {kf_mean:.4f} ± {kf_std:.4f}")

    gkf_mean, gkf_std = nested_r2(X_raw, y_log, groups, xgb_pipeline, grid,
                                  GroupKFold(n_splits=N_SPLITS), True)
    print(f"GroupKFold R2     : {gkf_mean:.4f} ± {gkf_std:.4f}")

    write_table(
        pd.DataFrame([
            {"Validation": "Standard KFold", "R-squared": f"{kf_mean:.4f} ± {kf_std:.4f}"},
            {"Validation": "GroupKFold", "R-squared": f"{gkf_mean:.4f} ± {gkf_std:.4f}"},
        ]),
        "Sensitivity_LogModel_CV",
    )

    model = fit_full_model(X, y_log)
    values = shap_matrix(model, X)

    # Figure 8: importance ranking of the log-outcome model.
    plt.figure(figsize=(10, 8))
    shap.summary_plot(values, X, plot_type="bar", show=False, color=COLOR_PRIMARY)
    plt.title("Feature Importance (Log-Transformed CO2 Model)", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Mean |SHAP Value|", fontsize=12)
    plt.tight_layout()
    save_figure(plt.gcf(), "Figure8_LogModel_Feature_Importance")
    plt.close(plt.gcf())

    # Figure 9: GDP dependence, rescaled to 100 million RMB.
    X_plot = X.copy()
    X_plot["GDP_Total"] = X_plot["GDP_Total"] / TO_100M_RMB
    plt.figure(figsize=(8, 6))
    shap.dependence_plot("GDP_Total", values, X_plot, interaction_index=None, show=False)
    plt.title("SHAP Dependence: Total GDP (Log CO2 Model)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Total GDP (100 Million Yuan)", fontsize=12)
    plt.tight_layout()
    save_figure(plt.gcf(), "Figure9_LogModel_SHAP_Dependence_GDP")
    plt.close(plt.gcf())

    # Figure 10: industrial electricity ratio dependence.
    plt.figure(figsize=(8, 6))
    shap.dependence_plot("Ind_Electricity_Ratio", values, X, interaction_index=None, show=False)
    plt.title("SHAP Dependence: Ind. Elec. Ratio (Log CO2 Model)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Industrial Electricity Ratio (%)", fontsize=12)
    plt.tight_layout()
    save_figure(plt.gcf(), "Figure10_LogModel_SHAP_Dependence_IndElectricity")
    plt.close(plt.gcf())

    print("[saved] results/Figure8_*, Figure9_*, Figure10_* (pdf/png)")


if __name__ == "__main__":
    main()
