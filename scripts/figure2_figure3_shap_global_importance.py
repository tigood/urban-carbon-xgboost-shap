"""Figures 2 and 3 - Global SHAP feature importance and summary distribution.

Figure 2 ranks predictors by their mean absolute SHAP value, giving the
magnitude of each variable's contribution to the model output. Figure 3 shows
the full distribution of SHAP values per variable, coloured by the observed
feature value, so that both direction and spread of the effects are visible.

Both are computed from the full-sample XGBoost model configured with the
optimal hyperparameters reported in Section 2.5 of the manuscript.

Outputs: results/Figure2_Feature_Importance_SHAP.pdf / .png
         results/Figure3_SHAP_Summary_Beeswarm.pdf / .png
"""

import matplotlib.pyplot as plt
import shap

from src.config import COLOR_PRIMARY, RANDOM_STATE, apply_journal_style, save_figure
from src.explain import analysis_sample, fit_full_model, shap_matrix
from src.preprocessing import load_panel


def main() -> None:
    apply_journal_style()

    panel = load_panel()
    X, y = analysis_sample(panel)
    model = fit_full_model(X, y)
    values = shap_matrix(model, X)

    # Figure 2: ranked mean absolute SHAP values.
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        values, X, plot_type="bar", show=False, color=COLOR_PRIMARY, plot_size=(10, 8)
    )
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=12)
    plt.tight_layout()
    save_figure(plt.gcf(), "Figure2_Feature_Importance_SHAP")
    plt.close(plt.gcf())

    # Figure 3: direction and distribution of each feature's contribution.
    # The beeswarm jitter is driven by NumPy's global RNG unless a generator is
    # supplied, so rng is passed explicitly to make the layout reproducible.
    shap.summary_plot(
        values, X, show=False, cmap="coolwarm", plot_size=(10, 8), rng=RANDOM_STATE
    )
    plt.title(
        "SHAP Summary Plot (Direction and Distribution of Impact)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("SHAP Value (Impact on CO2 Emission)", fontsize=12)
    plt.tight_layout()
    save_figure(plt.gcf(), "Figure3_SHAP_Summary_Beeswarm")
    plt.close(plt.gcf())

    print("[saved] results/Figure2_*.pdf/.png, results/Figure3_*.pdf/.png")


if __name__ == "__main__":
    main()
