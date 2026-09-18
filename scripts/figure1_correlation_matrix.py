"""Figure 1 - Pearson correlation matrix of the selected variables.

Winsorized predictor correlations are shown as a lower-triangle heatmap with
the outcome in the leading row/column. The strong correlations between the
macro-scale variables (Total GDP with government expenditure and public buses)
are reported in the manuscript and motivate the joint interpretation caveat in
Section 3.4.

Output: results/Figure1_Correlation_Matrix.pdf / .png
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import TARGET, apply_journal_style, save_figure
from src.preprocessing import load_panel, winsorize_frame

IDENTIFIERS = ["City", "Year"]


def main() -> None:
    apply_journal_style()

    panel = load_panel()
    # The outcome is part of the matrix, so build the frame directly rather
    # than through the modelling feature matrix.
    frame = winsorize_frame(panel.drop(columns=IDENTIFIERS).astype(float))

    # Keep the outcome first so the heatmap reads top-left to bottom-right.
    columns = [TARGET] + [c for c in frame.columns if c != TARGET]
    corr = frame[columns].corr(method="pearson")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        annot_kws={"size": 9},
        cbar_kws={"label": "Pearson Correlation Coefficient", "shrink": 0.8},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(
        "Pearson Correlation Matrix of Selected Variables",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    save_figure(fig, "Figure1_Correlation_Matrix")
    plt.close(fig)
    print("[saved] results/Figure1_Correlation_Matrix.pdf / .png")


if __name__ == "__main__":
    main()
