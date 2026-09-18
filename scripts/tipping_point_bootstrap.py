"""Empirical tipping points of Total GDP and the industrial electricity ratio.

A segmented (two-piece) regression is fitted to the SHAP dependence data. For
each candidate breakpoint inside the 15th-85th percentile range the sum of
squared errors of an independent linear fit on either side is computed, and
the breakpoint minimising that total is retained. The search range is trimmed
to leave a sufficient number of observations on both sides and to avoid
spurious edge effects.

Stability is assessed by a city-level bootstrap: the 231 cities are resampled
with replacement 1,000 times and the segmented regression is repeated on the
resampled SHAP values. Because the bootstrap resamples previously computed
SHAP values rather than refitting XGBoost, the resulting intervals describe
the spatial sampling uncertainty of the estimated relationship, not the
model-fitting uncertainty. This distinction is stated in Section 2.6.

Output: results/Tipping_Points_Bootstrap.csv
"""

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, TO_100M_RMB, write_table
from src.explain import analysis_sample, fit_full_model, shap_matrix
from src.preprocessing import load_panel

N_ITERATIONS = 1000
TRIM_RATIO = 0.15
FEATURES = [("GDP_Total", TO_100M_RMB), ("Ind_Electricity_Ratio", 1.0)]


def find_breakpoint(x: np.ndarray, y: np.ndarray, trim_ratio: float = TRIM_RATIO) -> float:
    """Breakpoint that minimises the total SSE of two independent linear fits.

    Sorting is done once and the left/right fits are evaluated for every
    candidate split in the trimmed range.
    """
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    start, stop = int(n * trim_ratio), int(n * (1 - trim_ratio))

    best_sse, best_breakpoint = np.inf, None
    for i in range(start, stop):
        left = np.polyfit(x[:i], y[:i], 1)
        right = np.polyfit(x[i:], y[i:], 1)
        sse = np.sum((np.polyval(left, x[:i]) - y[:i]) ** 2)
        sse += np.sum((np.polyval(right, x[i:]) - y[i:]) ** 2)
        if sse < best_sse:
            best_sse, best_breakpoint = sse, x[i]

    return best_breakpoint


def main() -> None:
    panel = load_panel()
    X, y = analysis_sample(panel)
    cities = panel["City"].to_numpy()

    model = fit_full_model(X, y)
    values = shap_matrix(model, X)

    unique_cities = np.unique(cities)
    np.random.seed(RANDOM_STATE)

    # Pre-compute the row index of every city so resampling is a lookup.
    city_rows = {city: np.where(cities == city)[0] for city in unique_cities}

    columns = {feature: X.columns.get_loc(feature) for feature, _ in FEATURES}
    breakpoints = {feature: [] for feature, _ in FEATURES}
    print(f"[bootstrap] {N_ITERATIONS} city-level resamples over {len(unique_cities)} cities")

    for iteration in range(N_ITERATIONS):
        sampled = np.random.choice(unique_cities, size=len(unique_cities), replace=True)
        sampled_rows = np.concatenate([city_rows[city] for city in sampled])

        for feature, _ in FEATURES:
            column = columns[feature]
            breakpoints[feature].append(
                find_breakpoint(X.iloc[sampled_rows, column].to_numpy(), values[sampled_rows, column])
            )

        if (iteration + 1) % 100 == 0:
            print(f"  {iteration + 1}/{N_ITERATIONS}")

    results = []
    for feature, divisor in FEATURES:
        samples = np.asarray(breakpoints[feature]) / divisor
        median = np.median(samples)
        lower, upper = np.percentile(samples, [2.5, 97.5])
        unit = "100 Million Yuan" if divisor > 1 else "%"
        print(f"{feature}: median = {median:,.2f} {unit}, 95% CI = [{lower:,.2f}, {upper:,.2f}]")
        results.append({
            "Feature": feature,
            "Median": round(float(median), 2),
            "CI_lower": round(float(lower), 2),
            "CI_upper": round(float(upper), 2),
            "Unit": unit,
        })

    write_table(pd.DataFrame(results), "Tipping_Points_Bootstrap")


if __name__ == "__main__":
    main()
