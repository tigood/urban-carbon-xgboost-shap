"""Table S1 - XGBoost sensitivity using originally observed CO2 outcomes only.

The input panel retains the main experiment's 231-city universe and identical
predictor values, but removes city-year rows whose CO2 outcome was created by
the balanced-panel interpolation. Both nested-validation levels use city-based
GroupKFold, matching the grouped main-model benchmark.

Output: results/TableS1_CO2_observed_only_sensitivity.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.table2_table3_model_benchmark import (
    N_INNER_SPLITS,
    N_SPLITS,
    SCORING,
    nested_evaluate,
)
from src.config import DATA_DIR, GROUP_COL, TARGET, write_table
from src.models import PARAM_GRIDS, benchmark_models
from src.preprocessing import feature_matrix, load_panel


OBSERVED_PANEL_FILE = DATA_DIR / "city_carbon_panel_2011_2019_observed_CO2_only.csv"
EXPECTED_ROWS = 1811
EXPECTED_CITIES = 231
EXPECTED_FEATURES = 18


class SingleSplit:
    """Expose one precomputed outer fold through the CV splitter interface."""

    def __init__(self, train_idx, test_idx):
        self.train_idx = train_idx
        self.test_idx = test_idx

    def split(self, X, y=None, groups=None):
        yield self.train_idx, self.test_idx


def validate_panel(panel: pd.DataFrame) -> None:
    """Fail before model fitting if the observed-only sample is not as audited."""
    checks = {
        "rows": len(panel),
        "cities": panel[GROUP_COL].nunique(),
        "co2_missing": int(panel[TARGET].isna().sum()),
        "features": feature_matrix(panel).shape[1],
        "year_min": int(panel["Year"].min()),
        "year_max": int(panel["Year"].max()),
        "duplicate_city_year": int(panel.duplicated([GROUP_COL, "Year"]).sum()),
    }
    expected = {
        "rows": EXPECTED_ROWS,
        "cities": EXPECTED_CITIES,
        "co2_missing": 0,
        "features": EXPECTED_FEATURES,
        "year_min": 2011,
        "year_max": 2019,
        "duplicate_city_year": 0,
    }
    if checks != expected:
        raise RuntimeError(
            "Observed-only panel validation failed; experiment stopped. "
            f"Expected {expected}, found {checks}."
        )


def log_fold_results(X, y, groups) -> None:
    """Print each outer-fold result while delegating all fitting to nested_evaluate."""
    outer_cv = GroupKFold(n_splits=N_SPLITS)
    splits = list(outer_cv.split(X, y, groups))

    print("\n--- Outer-fold results ---")
    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        result = nested_evaluate(
            benchmark_models()["XGBoost"],
            PARAM_GRIDS["XGBoost"],
            X,
            y,
            SingleSplit(train_idx, test_idx),
            groups,
            grouped=True,
        )
        train_cities = groups.iloc[train_idx].nunique()
        test_cities = groups.iloc[test_idx].nunique()
        print(
            f"Fold {fold_number}: train={len(train_idx)} obs/{train_cities} cities, "
            f"test={len(test_idx)} obs/{test_cities} cities, "
            f"R2={result['R-squared'].split(' ± ')[0]}, "
            f"RMSE={result['RMSE'].split(' ± ')[0]}, "
            f"MAE={result['MAE'].split(' ± ')[0]}"
        )


def main() -> None:
    if not Path(OBSERVED_PANEL_FILE).exists():
        raise FileNotFoundError(f"Observed-only panel not found: {OBSERVED_PANEL_FILE}")

    panel = load_panel(OBSERVED_PANEL_FILE)
    validate_panel(panel)

    X = feature_matrix(panel)
    y = panel[TARGET].astype(float)
    groups = panel[GROUP_COL]

    print("\n=== Observed CO2 only sensitivity ===")
    print(f"Sample size       : {len(panel)}")
    print(f"Number of cities  : {panel[GROUP_COL].nunique()}")
    print(f"Number of features: {X.shape[1]}")
    print(f"CO2 missing       : {panel[TARGET].isna().sum()}")
    print(f"Outer CV          : GroupKFold({N_SPLITS}), group={GROUP_COL}")
    print(f"Inner CV          : GroupKFold({N_INNER_SPLITS})")
    print(f"Selection scoring : {SCORING}")

    log_fold_results(X, y, groups)

    print("\n--- Aggregate nested-CV result ---")
    aggregate = nested_evaluate(
        benchmark_models()["XGBoost"],
        PARAM_GRIDS["XGBoost"],
        X,
        y,
        GroupKFold(n_splits=N_SPLITS),
        groups,
        grouped=True,
    )
    result_table = pd.DataFrame(
        [
            {
                "Dataset": "Observed CO2 only",
                "Model": "XGBoost",
                "R2": aggregate["R-squared"],
                "RMSE": aggregate["RMSE"],
                "MAE": aggregate["MAE"],
            }
        ]
    )
    print(result_table.to_string(index=False))
    write_table(result_table, "TableS1_CO2_observed_only_sensitivity")


if __name__ == "__main__":
    main()
