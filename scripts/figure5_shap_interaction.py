"""Figure 5 - Joint SHAP dependence of Total GDP and government expenditure.

The dependence plot for Total GDP is coloured by local government
expenditure. Because the two variables are strongly correlated (r = 0.86 in
Figure 1) they share most of their variation, so the joint pattern is an
attribution under the observed sample rather than an independent interaction
effect. This caveat is carried explicitly in Section 3.4 of the manuscript.

Output: results/Figure5_SHAP_Interaction.pdf / .png
"""

import matplotlib.pyplot as plt
import shap

from src.config import TO_100M_RMB, apply_journal_style, save_figure
from src.explain import analysis_sample, fit_full_model, shap_matrix
from src.preprocessing import load_panel

FOCUS_FEATURE = "GDP_Total"
COLOR_FEATURE = "Gov_Expenditure"


def main() -> None:
    apply_journal_style()

    panel = load_panel()
    X, y = analysis_sample(panel)
    model = fit_full_model(X, y)
    values = shap_matrix(model, X)

    # Rescale the two fiscal-scale variables for the plot only.
    X_plot = X.copy()
    X_plot[FOCUS_FEATURE] = X_plot[FOCUS_FEATURE] / TO_100M_RMB
    X_plot[COLOR_FEATURE] = X_plot[COLOR_FEATURE] / TO_100M_RMB

    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        FOCUS_FEATURE,
        values,
        X_plot,
        interaction_index=COLOR_FEATURE,
        show=False,
        cmap="coolwarm",
    )
    plt.xlabel("Total GDP (100 Million Yuan)", fontsize=12)
    plt.ylabel("SHAP Value of Total GDP", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    save_figure(plt.gcf(), "Figure5_SHAP_Interaction")
    plt.close(plt.gcf())
    print("[saved] results/Figure5_SHAP_Interaction.pdf / .png")


if __name__ == "__main__":
    main()
