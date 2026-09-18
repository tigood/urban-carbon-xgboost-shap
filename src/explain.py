"""Fitting and explanation helpers shared by the SHAP scripts.

The attribution results reported in the manuscript are computed from a single
XGBoost model refitted on the complete sample with the optimal
hyperparameters, rather than from out-of-fold predictions. Every SHAP figure,
ranking and tipping-point estimate therefore refers to the same model object.
"""

import numpy as np
import pandas as pd
import shap

from src.models import final_xgb
from src.preprocessing import feature_matrix, target, winsorize_frame


def prepare_frame(df: pd.DataFrame, drop_buses: bool = False) -> pd.DataFrame:
    """Return the winsorized predictor frame used for a full-sample fit.

    Winsorization is estimated once on the full sample here; in the
    cross-validation scripts the same transformation is instead fitted inside
    each training fold.
    """
    return winsorize_frame(feature_matrix(df, drop_buses=drop_buses))


def fit_full_model(X: pd.DataFrame, y: pd.Series):
    """Fit the interpretation model on the complete sample."""
    model = final_xgb()
    model.fit(X, y)
    return model


def shap_matrix(model, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values for every observation-feature pair."""
    return shap.TreeExplainer(model).shap_values(X)


def mean_abs_shap(model, X: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per feature, i.e. the importance ranking."""
    values = np.abs(shap_matrix(model, X)).mean(axis=0)
    return pd.Series(values, index=X.columns).sort_values(ascending=False)


def analysis_sample(df: pd.DataFrame, drop_buses: bool = False):
    """Convenience wrapper returning the winsorized frame and the outcome."""
    return prepare_frame(df, drop_buses=drop_buses), target(df)
