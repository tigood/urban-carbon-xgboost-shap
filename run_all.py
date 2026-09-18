"""Reproduce every table and figure of the manuscript in order.

Usage (from the repository root):

    python run_all.py            # full reproduction
    python run_all.py --fast     # skip the two long-running steps

The two long-running steps are the nested cross-validation benchmark
(Tables 2 and 3) and the city-level bootstrap of the tipping points. Both are
correct but computationally heavy.
"""

import argparse
import time

from src.config import RESULTS_DIR

MODULES = [
    ("Table 1  - descriptive statistics", "scripts.table1_descriptive_statistics", False),
    ("Figure 1 - correlation matrix", "scripts.figure1_correlation_matrix", False),
    ("Tables 2/3 - model benchmark", "scripts.table2_table3_model_benchmark", True),
    ("Figures 2/3 - global SHAP", "scripts.figure2_figure3_shap_global_importance", False),
    ("Figure 4 - SHAP dependence", "scripts.figure4_shap_dependence", False),
    ("Figure 5 - SHAP interaction", "scripts.figure5_shap_interaction", False),
    ("Figure 6 - heating heterogeneity", "scripts.figure6_heating_heterogeneity", False),
    ("Figure 7 - ranking robustness", "scripts.figure7_hyperparameter_robustness", False),
    ("Figures 8-10 - log-outcome sensitivity", "scripts.figure8_10_log_model_sensitivity", True),
    ("Tipping points - bootstrap", "scripts.tipping_point_bootstrap", True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the nested cross-validation and bootstrap steps",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for label, module_name, slow in MODULES:
        if slow and args.fast:
            print(f"\n[skip] {label}")
            continue

        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
        started = time.time()
        module = __import__(module_name, fromlist=["main"])
        module.main()
        print(f"[done] {label} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
