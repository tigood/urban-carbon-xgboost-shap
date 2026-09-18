"""Figure 4 - SHAP dependence plots for the four core predictors.

Shows the marginal contribution of Total GDP, the industrial electricity
ratio, precipitation and government expenditure across their observed ranges.
The panels are ordered by mean absolute SHAP value, matching the importance
ranking in Figure 2. The two fiscal-scale variables are converted from 10,000
RMB to 100 million RMB for readability.

The non-linear transitions visible here are quantified formally - segmented
regression with a city-level bootstrap - in scripts/tipping_point_bootstrap.py.

Output: results/Figure4_SHAP_Dependence.pdf / .png
"""

import matplotlib.pyplot as plt

from src.config import COLOR_PRIMARY, TO_100M_RMB, apply_journal_style, save_figure
from src.explain import analysis_sample, fit_full_model, shap_matrix
from src.preprocessing import load_panel

# Predictors shown, in descending order of mean absolute SHAP value.
CORE_FEATURES = [
    ("GDP_Total", "Total GDP", " (100 Million Yuan)", TO_100M_RMB),
    ("Ind_Electricity_Ratio", "Industrial Electricity Ratio", " (%)", 1.0),
    ("Precipitation", "Precipitation", " (mm)", 1.0),
    ("Gov_Expenditure", "Government Expenditure", " (100 Million Yuan)", TO_100M_RMB),
]


def main() -> None:
    apply_journal_style()

    panel = load_panel()
    X, y = analysis_sample(panel)
    model = fit_full_model(X, y)
    values = shap_matrix(model, X)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, (feature, label, unit, divisor) in zip(axes.flatten(), CORE_FEATURES):
        x_values = X[feature].to_numpy() / divisor
        ax.scatter(x_values, values[:, X.columns.get_loc(feature)], alpha=0.6, color=COLOR_PRIMARY, s=15)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"SHAP Dependence: {label}", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"{label} Actual Value{unit}", fontsize=12)
        ax.set_ylabel("SHAP Value (Impact on CO2)", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout(pad=3.0)
    save_figure(fig, "Figure4_SHAP_Dependence")
    plt.close(fig)
    print("[saved] results/Figure4_SHAP_Dependence.pdf / .png")


if __name__ == "__main__":
    main()
