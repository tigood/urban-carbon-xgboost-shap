# Decoding Nonlinear Predictive Patterns in Urban Carbon Emissions

Reproduction code for the manuscript *"Decoding Nonlinear Predictive Patterns
in Urban Carbon Emissions: An Interpretable Machine Learning Framework based on
XGBoost and SHAP"* (Zhang & Liu, Heliyon).

The analysis fits an XGBoost model to a balanced panel of 231 Chinese cities
over 2011–2019, interprets it with SHAP, and quantifies non-linear structural
shifts through segmented regression with a city-level bootstrap. Two
reviewer-requested supplementary analyses test whether the main conclusions
depend on interpolated CO2 outcomes or on the high correlation between total
GDP and government expenditure.

## Repository layout

```
├── data/
│   ├── README.md                              data provenance, units, construction
│   ├── city_carbon_panel_2011_2019.csv        processed analysis panel (2,079 x 21)
│   └── city_carbon_panel_2011_2019_observed_CO2_only.csv
│                                                observed-CO2 subset (1,811 x 21)
├── scripts/                                   manuscript and supplementary artifacts
│   ├── table1_descriptive_statistics.py
│   ├── figure1_correlation_matrix.py
│   ├── table2_table3_model_benchmark.py
│   ├── figure2_figure3_shap_global_importance.py
│   ├── figure4_shap_dependence.py
│   ├── figure5_shap_interaction.py
│   ├── figure6_heating_heterogeneity.py
│   ├── figure7_hyperparameter_robustness.py
│   ├── figure8_10_log_model_sensitivity.py
│   ├── sensitivity_co2_observed_only.py
│   ├── sensitivity_shap_without_gov_expenditure.py
│   └── tipping_point_bootstrap.py
├── src/
│   ├── config.py                              paths, constants, figure styling
│   ├── preprocessing.py                       data loading and the Winsorizer transformer
│   ├── models.py                              benchmark specifications and search spaces
│   └── explain.py                             full-sample fit and SHAP helpers
├── results/                                   generated outputs (tables as CSV, figures as PDF + PNG)
├── run_all.py                                 runs every script in order
└── requirements.txt
```

## Environment

Python 3.12.4. Create an isolated environment and install the pinned
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

All commands are issued from the repository root, so that `src` is importable
as a package.

```bash
python run_all.py                  # full reproduction
python run_all.py --fast           # skip computationally heavy CV/bootstrap steps
```

Individual artifacts can be produced on their own, for example:

```bash
python -m scripts.figure4_shap_dependence
python -m scripts.sensitivity_co2_observed_only
python -m scripts.sensitivity_shap_without_gov_expenditure
```

Outputs are written to `results/`. Figures are saved in both PDF (vector, for
typesetting) and PNG at 600 dpi (for review).

Runtime on a standard desktop: the per-figure SHAP scripts take a few seconds
each; the nested cross-validation benchmark runs for roughly 15–30 minutes
because the inner grid search is repeated inside every outer fold; the
observed-CO2 sensitivity repeats the same nested design and is therefore also
computationally intensive; the 1,000-iteration bootstrap takes roughly 5–10
minutes. The bootstrap writes its progress every 100 iterations, so it is
worth running with unbuffered output
(`python -u -m scripts.tipping_point_bootstrap`) if you want to watch it.

## Mapping from scripts to manuscript artifacts

| Artifact | Script | Notes |
| --- | --- | --- |
| Table 1 | `table1_descriptive_statistics.py` | 1st/99th percentile winsorized descriptives, shown in the units used in the paper |
| Figure 1 | `figure1_correlation_matrix.py` | Lower-triangle Pearson matrix of the winsorized predictors |
| Table 2 | `table2_table3_model_benchmark.py` | 5-fold KFold, nested hyperparameter search |
| Table 3 | `table2_table3_model_benchmark.py` | 5-fold GroupKFold by city, same nested design |
| Figure 2 | `figure2_figure3_shap_global_importance.py` | Mean absolute SHAP ranking |
| Figure 3 | `figure2_figure3_shap_global_importance.py` | SHAP summary beeswarm plot |
| Figure 4 | `figure4_shap_dependence.py` | Dependence panels for the four core predictors |
| Figure 5 | `figure5_shap_interaction.py` | GDP dependence coloured by government expenditure |
| Figure 6 | `figure6_heating_heterogeneity.py` | Heating versus non-heating city subgroups |
| Figure 7 | `figure7_hyperparameter_robustness.py` | Rank stability across `max_depth` ∈ {2, 4, 6} |
| Figures 8–10 | `figure8_10_log_model_sensitivity.py` | Log-outcome model with `Buses` removed |
| Table S1 | `sensitivity_co2_observed_only.py` | Nested city-grouped XGBoost evaluation using only originally observed CO2 outcomes |
| Supplementary SHAP sensitivity | `sensitivity_shap_without_gov_expenditure.py` | Full versus reduced SHAP rankings after removing `Gov_Expenditure` |
| Tipping points | `tipping_point_bootstrap.py` | Segmented regression, 1,000 city-level resamples |

## Reviewer-requested sensitivity analyses

**Originally observed CO2 outcomes only.** The balanced analysis panel contains
2,079 city-year rows, whereas 1,811 rows have an originally measured CO2
outcome. `sensitivity_co2_observed_only.py` restricts the analysis to those
1,811 rows while retaining all 231 cities, the 2011–2019 study window and the
same 18 predictors. The script reuses the main XGBoost pipeline and nested
city-grouped validation design: five outer folds, three inner folds and
negative RMSE for hyperparameter selection. Its aggregate result is R2 =
0.3595 ± 0.1386, RMSE = 36.2621 ± 9.5032 and MAE = 21.0491 ± 2.8247. For
comparison, the main grouped XGBoost result is R2 = 0.3941 ± 0.1279, RMSE =
34.2757 ± 8.1152 and MAE = 20.3076 ± 2.5010. The modest change indicates that
the predictive conclusion is not driven solely by interpolated outcome rows.
The result is written to
`results/TableS1_CO2_observed_only_sensitivity.csv`.

**SHAP sensitivity without government expenditure.**
`sensitivity_shap_without_gov_expenditure.py` fits the baseline 18-feature
interpretation model and a 17-feature model that removes only
`Gov_Expenditure`. Both use the main panel, identical full-sample
Winsorization, the fixed XGBoost interpretation parameters and exact TreeSHAP.
Across the 17 common predictors, the mean absolute SHAP ranks have Spearman ρ
= 0.9485 (p = 6.86 × 10^-9). `GDP_Total` remains ranked first, and its mean
absolute SHAP value changes from 13.0729 to 13.7539 (+5.21%). The global
ranking and the central interpretation of total GDP are therefore stable to
excluding the correlated fiscal predictor, although some middle-ranked
features change position. The script writes ranking and summary CSV files plus
the reduced-model importance and GDP dependence-comparison figures to
`results/`.

## Method notes

**Leakage-free preprocessing.** Winsorization is implemented as a scikit-learn
transformer (`src/preprocessing.py`). Its percentile bounds are estimated in
`fit`, so inside a cross-validation run they are derived only from the training
portion of each split. Binary indicators are detected by cardinality and are
never winsorized. Standard scaling for the OLS and SVR specifications is
enclosed in the same pipeline. The outcome variable is never winsorized during
model fitting, so its true maximum of 457.76 Mt is preserved.

**Nested cross-validation.** Performance figures in Tables 2 and 3 come from a
true nested design. A 5-fold outer loop is reserved exclusively for evaluation.
Inside each outer training fold, a separate 3-fold inner search selects
hyperparameters by negative RMSE, and the search is repeated independently for
every outer fold. Under the grouped regime both loops use `GroupKFold` on the
city identifier, so a city never straddles a fold boundary at either level.
Table S1 applies the same grouped design and scoring rule to the observed-only
outcome subset; its only sample change is the removal of city-year rows whose
CO2 outcome was not originally observed.

**Source of the SHAP values.** All attribution results — importance rankings,
dependence plots, tipping points and subgroup comparisons — are computed from a
single XGBoost model refitted on the complete sample with the optimal
hyperparameters returned by the inner search (100 trees, learning rate 0.05,
maximum depth 4). The same configuration underlies Figures 2–6, so those panels
are mutually consistent. Figure 7 additionally refits two shallower and deeper
variants for the ranking-stability check, and Figures 8–10 use the same
configuration on the log-transformed outcome. SHAP values are not cross-fitted,
and the manuscript states this in Section 2.4. The supplementary correlated-
feature sensitivity uses the same full-sample fitting and TreeSHAP procedure
for both specifications, changing only the presence of `Gov_Expenditure`.

**Tipping points.** The breakpoint is the split minimising the summed squared
error of two independent linear fits, searched over the 15th–85th percentile
range of the feature. Stability comes from 1,000 bootstrap resamples of the 231
cities. Because the bootstrap resamples previously computed SHAP values rather
than refitting XGBoost, the intervals express spatial sampling uncertainty and
not model-fitting uncertainty, as noted in Section 2.6.

**Regional comparison.** Figure 6 compares heating and non-heating cities using
SHAP values from the single global model, not from separately fitted subgroup
models, so both groups share a common baseline. Heating status also proxies
climate, energy infrastructure and industrial structure; without a formal
interaction test, the contrast remains exploratory.

## Reproducibility

A fixed `random_state = 42` is used for the train/validation splits, the model
initialisation and the bootstrap. The scripts were last verified to reproduce
Tables 2 and 3, the observed-only Table S1 metrics, the correlated-feature SHAP
ranking comparison, the sensitivity-model R-squared values and the reported
tipping-point intervals.

One rendering detail is worth noting: SHAP's summary beeswarm draws its
vertical jitter from NumPy's global random state, so Figure 3 is not
reproducible between runs unless a generator is supplied. A seeded generator is
passed explicitly here, which makes the figure stable across runs without
changing any plotted SHAP value. All other figures are deterministic.

## Citation and contact

Zhang, H.; Liu, S. *Decoding Nonlinear Predictive Patterns in Urban Carbon
Emissions: An Interpretable Machine Learning Framework based on XGBoost and
SHAP.* Heliyon.

Correspondence: zhanghaoit@ncst.edu.cn
