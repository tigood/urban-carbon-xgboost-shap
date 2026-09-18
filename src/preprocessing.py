"""Data loading and leakage-free preprocessing.

Winsorization is implemented as a scikit-learn transformer rather than as a
one-off operation on the full sample. The percentile bounds are estimated in
``fit`` and are therefore derived only from the training portion of each
cross-validation split, which keeps the preprocessing pipeline leakage-free.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    EXCLUDE_FROM_FEATURES,
    PANEL_FILE,
    TARGET,
    WINSOR_LOWER,
    WINSOR_UPPER,
)


def load_panel(path=PANEL_FILE) -> pd.DataFrame:
    """Load the balanced city panel used throughout the study."""
    df = pd.read_csv(path, encoding="utf-8")
    print(f"[data] {path.name}: {df.shape[0]} observations, {df.shape[1]} columns")
    return df


def feature_matrix(df: pd.DataFrame, drop_buses: bool = False) -> pd.DataFrame:
    """Return the continuous/indicator predictors used by the tree models.

    ``drop_buses`` removes the public-bus count, which is used only in the
    log-outcome sensitivity analysis (Section 4.3 of the paper).
    """
    excluded = list(EXCLUDE_FROM_FEATURES)
    if drop_buses:
        excluded.append("Buses")
    return df.drop(columns=[c for c in excluded if c in df.columns]).astype(float)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip continuous features to the [lower, upper] percentile range.

    Features with two or fewer distinct values are treated as binary
    indicators and are passed through unchanged, so that dummy variables are
    not rescaled. The clipping bounds are stored as attributes of the fitted
    transformer and travel with it inside a scikit-learn ``Pipeline``.
    """

    def __init__(self, lower: float = WINSOR_LOWER, upper: float = WINSOR_UPPER):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.numeric_cols_ = [c for c in frame.columns if frame[c].nunique() > 2]
        if self.numeric_cols_:
            self.limits_ = frame[self.numeric_cols_].quantile([self.lower, self.upper]).to_dict()
        else:
            self.limits_ = {}
        return self

    def transform(self, X):
        is_frame = isinstance(X, pd.DataFrame)
        frame = X.copy() if is_frame else pd.DataFrame(X).copy()
        for col in self.numeric_cols_:
            if col in self.limits_:
                frame[col] = frame[col].clip(
                    lower=self.limits_[col][self.lower],
                    upper=self.limits_[col][self.upper],
                )
        return frame if is_frame else frame.values


def winsorize_frame(df: pd.DataFrame, lower: float = WINSOR_LOWER, upper: float = WINSOR_UPPER) -> pd.DataFrame:
    """Winsorize a feature frame in place, keeping column names.

    Used by the scripts that need an explicit winsorized frame to feed SHAP
    (the tree models themselves are fitted on this frame).
    """
    return Winsorizer(lower=lower, upper=upper).fit_transform(df)


def target(df: pd.DataFrame) -> pd.Series:
    """Return the (never winsorized) emission outcome."""
    return df[TARGET].astype(float)
