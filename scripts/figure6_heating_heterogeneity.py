"""Figure 6 - Regional heterogeneity by centralised winter heating status.

Compares the SHAP dependence of Total GDP and government expenditure between
cities covered by the northern centralised heating policy and cities outside
it. The SHAP values of both subgroups come from the single global model, so
the two groups share the same baseline prediction and remain directly
comparable. The grouping is a regional marker that also proxies climate,
energy infrastructure and industrial structure; without a formal interaction
test the contrast is exploratory, as stated in Section 4.1.

Output: results/Figure6_Heating_Heterogeneity.pdf / .png
"""

import matplotlib.pyplot as plt

from src.config import COLOR_PRIMARY, COLOR_SECONDARY, apply_journal_style, save_figure
from src.explain import analysis_sample, fit_full_model, shap_matrix
from src.preprocessing import load_panel

HEATING_FLAG = "Is_Heating_City"
COMPARED_FEATURES = [
    ("GDP_Total", "Total GDP"),
    ("Gov_Expenditure", "Government Expenditure"),
]


def main() -> None:
    apply_journal_style()

    panel = load_panel()
    X, y = analysis_sample(panel)
    model = fit_full_model(X, y)
    values = shap_matrix(model, X)

    heating = X[HEATING_FLAG] == 1
    print(f"[sample] heating cities: {int(heating.sum())} obs, non-heating: {int((~heating).sum())} obs")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (feature, label) in zip(axes, COMPARED_FEATURES):
        x_values = X[feature].to_numpy()
        feature_shap = values[:, X.columns.get_loc(feature)]

        ax.scatter(
            x_values[~heating], feature_shap[~heating],
            alpha=0.5, color=COLOR_PRIMARY, s=20, label="Non-Heating Cities (South)",
        )
        ax.scatter(
            x_values[heating], feature_shap[heating],
            alpha=0.6, color=COLOR_SECONDARY, s=20, label="Heating Cities (North)",
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Heterogeneity by Region: {label}", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"{label} Actual Value", fontsize=12)
        ax.set_ylabel("SHAP Value (Impact on CO2)", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    save_figure(fig, "Figure6_Heating_Heterogeneity")
    plt.close(fig)
    print("[saved] results/Figure6_Heating_Heterogeneity.pdf / .png")


if __name__ == "__main__":
    main()
