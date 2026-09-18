"""Model definitions and hyperparameter search spaces.

All estimators are wrapped in a ``Pipeline`` whose first step is the
``Winsorizer``, so the preprocessing bounds are re-estimated inside every
training fold.
"""

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.config import RANDOM_STATE, XGB_FINAL_PARAMS
from src.preprocessing import Winsorizer


def winsorized(estimator) -> Pipeline:
    """Wrap an estimator so winsorization is fitted on the training fold only."""
    return Pipeline([("winsor", Winsorizer()), ("model", estimator)])


def twfe_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for the two-way fixed-effects OLS baseline.

    Adds city and year indicators so the linear benchmark can absorb
    unobserved city heterogeneity and common time shocks, matching the
    specification reported in Table 2.
    """
    return pd.get_dummies(
        df.drop(columns=["CO2_Emission"]),
        columns=["Year", "City"],
        drop_first=True,
        dtype=float,
    )


def benchmark_models() -> dict:
    """The four benchmark specifications compared in Tables 2 and 3."""
    return {
        "OLS (Two-way Fixed Effects)": Pipeline(
            [
                ("winsor", Winsorizer()),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "SVR (RBF Kernel)": Pipeline(
            [
                ("winsor", Winsorizer()),
                ("scaler", StandardScaler()),
                ("model", SVR()),
            ]
        ),
        "Random Forest": winsorized(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        "XGBoost": winsorized(xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    }


# Inner-loop search spaces. Percentile-free (no preprocessing) and deliberately
# narrow, since the inner loop runs inside every outer training fold.
PARAM_GRIDS = {
    "XGBoost": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [4, 6],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20],
    },
    "SVR (RBF Kernel)": {
        "model__C": [10.0, 50.0],
        "model__gamma": ["scale", 0.1],
    },
}


def final_xgb(**overrides) -> xgb.XGBRegressor:
    """XGBoost configured with the optimal hyperparameters used for SHAP.

    These are the optima returned by the inner cross-validation search, refit
    on the full sample so that all attribution results share one model.
    """
    params = {**XGB_FINAL_PARAMS, "random_state": RANDOM_STATE, "n_jobs": -1}
    params.update(overrides)
    return xgb.XGBRegressor(**params)
