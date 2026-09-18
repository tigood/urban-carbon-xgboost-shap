"""Project-wide paths, constants and figure defaults.

Every script in ``scripts/`` imports its paths and shared settings from here so
that the analysis can be relocated or re-run without editing individual files.
"""

from pathlib import Path

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Balanced city panel, 231 Chinese cities x 9 years (2011-2019).
PANEL_FILE = DATA_DIR / "city_carbon_panel_2011_2019.csv"

# Reproducibility
RANDOM_STATE = 42

# Winsorization bounds applied to continuous predictors inside the
# cross-validation pipeline. Binary indicators are never winsorized.
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

# Column roles in the panel
TARGET = "CO2_Emission"
GROUP_COL = "City"
YEAR_COL = "Year"
BINARY_COLS = ["Is_Resource_City", "Is_Heating_City"]
EXCLUDE_FROM_FEATURES = ["City", "Year", TARGET]

# Hyperparameters of the interpretation model. These are the optima returned by
# the inner cross-validation search of scripts/table2_table3_model_benchmark.py
# and are reused wherever SHAP values are computed on the full sample.
XGB_FINAL_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 4,
}

# Full-sample panel size, used for reporting the subgroup split in Figure 6.
N_CITIES = 231
N_PERIODS = 9

# Unit conversions applied for display only. The panel stores GDP and fiscal
# variables in 10,000 RMB, so dividing by 10^4 yields 100 million RMB.
TO_100M_RMB = 1e4

# Colour palette shared by all figures.
COLOR_PRIMARY = "#4A90E2"
COLOR_SECONDARY = "#E94E77"
COLOR_NEUTRAL = "#9B9B9B"


def apply_journal_style() -> None:
    """Set matplotlib defaults for journal submission figures."""
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["savefig.bbox"] = "tight"


def save_figure(fig, stem: str, dpi: int = 600) -> None:
    """Write a figure to ``results/`` in both vector and raster formats."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / f"{stem}.pdf", format="pdf")
    fig.savefig(RESULTS_DIR / f"{stem}.png", format="png", dpi=dpi)


def write_table(df, stem: str) -> None:
    """Write a results table to ``results/`` as UTF-8 CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{stem}.csv", index=False, encoding="utf-8")
    print(f"[saved] results/{stem}.csv")
