"""Figure 7 - Stability of the feature importance ranking across tree depths.

Three XGBoost models of increasing complexity (max_depth = 2, 4, 6) are fitted
on the same winsorized sample. For each model the predictors are ranked by
mean absolute SHAP value, and the resulting ranks are traced across the three
settings. A stable ranking indicates that the importance ordering is not an
artefact of a single tree depth.

Output: results/Figure7_Robustness_Ranking.pdf / .png
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import apply_journal_style, save_figure
from src.explain import analysis_sample, mean_abs_shap
from src.models import final_xgb
from src.preprocessing import load_panel

DEPTHS = [("Shallow Tree\n(max_depth=2)", 2), ("Baseline Model\n(max_depth=4)", 4), ("Deep Tree\n(max_depth=6)", 6)]
TOP_N = 8
LINE_COLORS = ["#E94E77", "#4A90E2", "#50E3C2", "#F5A623", "#9B9B9B", "#8B572A", "#7ED321", "#BD10E0"]


def main() -> None:
    apply_journal_style()

    panel = load_panel()
    X, y = analysis_sample(panel)

    ranks = {}
    for label, depth in DEPTHS:
        model = final_xgb(max_depth=depth)
        model.fit(X, y)
        ranks[label] = mean_abs_shap(model, X).rank(ascending=False, method="min")

    rank_table = pd.DataFrame(ranks)
    top_features = rank_table["Baseline Model\n(max_depth=4)"].sort_values().head(TOP_N).index
    plotted = rank_table.loc[top_features]

    fig, ax = plt.subplots(figsize=(10, 7))
    positions = range(len(DEPTHS))

    for idx, feature in enumerate(plotted.index):
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        ax.plot(positions, plotted.loc[feature], marker="o", markersize=10,
                linewidth=3, color=color, alpha=0.8, label=feature)
        ax.text(len(DEPTHS) - 1 + 0.05, plotted.loc[feature].iloc[-1], feature,
                ha="left", va="center", fontsize=11, fontweight="bold", color=color)

    max_rank = int(plotted.max().max())
    ax.set_ylim(max_rank + 0.5, 0.5)
    ax.set_yticks(range(1, max_rank + 1))
    ax.set_yticklabels(range(1, max_rank + 1), fontsize=11)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([label for label, _ in DEPTHS], fontsize=12, fontweight="bold")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title(
        "Robustness of Feature Importance Rankings Across Model Hyperparameters",
        fontsize=14, fontweight="bold", pad=20,
    )

    fig.tight_layout()
    save_figure(fig, "Figure7_Robustness_Ranking")
    plt.close(fig)
    print("[saved] results/Figure7_Robustness_Ranking.pdf / .png")


if __name__ == "__main__":
    main()
