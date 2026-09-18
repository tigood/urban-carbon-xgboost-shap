"""Tables 2 and 3 - Model performance under nested cross-validation.

Two evaluation regimes are run over the same four specifications:

* Table 2 uses standard 5-fold KFold, which measures in-sample interpolation
  among cities already seen during training.
* Table 3 uses 5-fold GroupKFold split by city identifier, which measures
  transferability to cities that never appear in the training folds.

In both regimes hyperparameters are selected in a strictly nested manner: the
outer folds are reserved for evaluation only, and an inner 3-fold search is
performed independently inside every outer training fold. The inner folds
follow the same grouping rule as the outer loop, so the city isolation
guarantee holds at both levels and the test folds stay unobserved during
tuning. Negative RMSE is the inner-loop selection metric.

The script also re-runs the search on the complete sample to report the global
optimal hyperparameters that are reused for the SHAP analysis.

Outputs: results/Table2_Model_Benchmark_KFold.csv
         results/Table3_Model_Benchmark_GroupKFold.csv
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold

from src.config import GROUP_COL, RANDOM_STATE, TARGET, write_table
from src.models import PARAM_GRIDS, benchmark_models, twfe_matrix
from src.preprocessing import feature_matrix, load_panel

N_SPLITS = 5
N_INNER_SPLITS = 3
SCORING = "neg_root_mean_squared_error"


def nested_evaluate(pipeline, param_grid, X, y, outer_cv, groups, grouped: bool) -> dict:
    """Evaluate one specification with tuning nested inside each outer fold.

    ``param_grid`` is None for the OLS baseline, which has no hyperparameters
    to tune. Returns mean and standard deviation across the outer folds.
    """
    r2_scores, rmse_scores, mae_scores = [], [], []

    # KFold ignores the groups argument, so only pass it under GroupKFold.
    splits = outer_cv.split(X, y, groups) if grouped else outer_cv.split(X, y)
    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if param_grid is None:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
        else:
            inner_cv = (
                GroupKFold(n_splits=N_INNER_SPLITS)
                if grouped
                else KFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            )
            search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring=SCORING, n_jobs=-1)
            if grouped:
                search.fit(X_train, y_train, groups=groups.iloc[train_idx])
            else:
                search.fit(X_train, y_train)
            predictions = search.best_estimator_.predict(X_test)

        r2_scores.append(r2_score(y_test, predictions))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, predictions)))
        mae_scores.append(mean_absolute_error(y_test, predictions))

    return {
        "R-squared": f"{np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}",
        "RMSE": f"{np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}",
        "MAE": f"{np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}",
        "_mean_r2": float(np.mean(r2_scores)),
    }


def run_regime(panel: pd.DataFrame, name: str, outer_cv, grouped: bool) -> pd.DataFrame:
    """Run all four specifications under one cross-validation regime."""
    models = benchmark_models()
    X_tree = feature_matrix(panel)
    X_twfe = twfe_matrix(panel)
    y = panel[TARGET].astype(float)
    groups = panel[GROUP_COL]

    specifications = [
        ("OLS (Two-way Fixed Effects)", models["OLS (Two-way Fixed Effects)"], None, X_twfe),
        ("XGBoost", models["XGBoost"], PARAM_GRIDS["XGBoost"], X_tree),
        ("Random Forest", models["Random Forest"], PARAM_GRIDS["Random Forest"], X_tree),
        ("SVR (RBF Kernel)", models["SVR (RBF Kernel)"], PARAM_GRIDS["SVR (RBF Kernel)"], X_tree),
    ]

    print(f"\n=== {name} ===")
    rows = []
    for label, pipeline, grid, X in specifications:
        print(f"  evaluating {label} ...")
        rows.append({"Model": label, **nested_evaluate(pipeline, grid, X, y, outer_cv, groups, grouped)})

    table = pd.DataFrame(rows).sort_values("_mean_r2", ascending=False).drop(columns="_mean_r2")
    print(table.to_string(index=False))
    return table


def report_global_optima(panel: pd.DataFrame) -> None:
    """Re-run the search on the full sample for the parameters used in SHAP.

    The resulting configuration is the one recorded in ``src.config`` as
    ``XGB_FINAL_PARAMS``; it governs the full-sample model behind Figures 2-7.
    """
    print("\n=== Global optimal hyperparameters (full sample, GroupKFold) ===")
    models = benchmark_models()
    X_tree = feature_matrix(panel)
    y = panel[TARGET].astype(float)
    groups = panel[GROUP_COL]

    for label in ["XGBoost", "Random Forest", "SVR (RBF Kernel)"]:
        search = GridSearchCV(
            models[label],
            PARAM_GRIDS[label],
            cv=GroupKFold(n_splits=N_SPLITS),
            scoring=SCORING,
            n_jobs=-1,
        )
        search.fit(X_tree, y, groups=groups)
        tuned = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}
        print(f"  {label}: {tuned}")


def main() -> None:
    panel = load_panel()

    table2 = run_regime(
        panel,
        "Table 2 (standard KFold)",
        KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        grouped=False,
    )
    write_table(table2, "Table2_Model_Benchmark_KFold")

    table3 = run_regime(
        panel, "Table 3 (GroupKFold by city)", GroupKFold(n_splits=N_SPLITS), grouped=True
    )
    write_table(table3, "Table3_Model_Benchmark_GroupKFold")

    report_global_optima(panel)


if __name__ == "__main__":
    main()
