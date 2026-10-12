# Data

## Files

`city_carbon_panel_2011_2019.csv` — balanced city-year panel, 2,079 observations
(231 cities × 9 years, 2011–2019), 21 columns, no missing values.

`city_carbon_panel_2011_2019_observed_CO2_only.csv` — sensitivity-analysis
subset, 1,811 observations, the same 231-city universe, the same 21 columns and
no missing CO2 outcomes. The panel is intentionally unbalanced because only
city-year rows with an originally measured CO2 outcome are retained.

## Provenance

| Block | Variables | Source |
| --- | --- | --- |
| Carbon emissions | `CO2_Emission` | CEADs city-level CO2 emission inventory, 1997–2019 |
| Socioeconomic | `GDP_Total`, `GDP_Per_Capita`, `Sec_Industry_Ratio`, `Ter_Industry_Ratio`, `Population`, `FDI`, `Gov_Expenditure`, `Tech_Expenditure`, `Buses`, `Urbanization_Rate`, `Ind_Electricity_Ratio` | China City Statistical Yearbook; EPS Data Platform |
| Urban form and environment | `Built_up_Area`, `Green_Coverage` | China City Statistical Yearbook |
| Climate | `HDD18`, `CDD26`, `Precipitation` | National Meteorological Science Data Center of China |
| Policy indicators | `Is_Resource_City`, `Is_Heating_City` | Resource-based city list; centralised winter heating policy (Qinling–Huaihe line) |

## Construction

The panel starts from 256 cities. Twenty-five cities with fewer than five years
of valid emission records were dropped, leaving 231 cities. A balanced
city-year grid was then constructed. Missing values, including missing
`CO2_Emission` outcomes and explanatory variables, were imputed by linear
interpolation within each city's own time series, using observations from both
temporal directions; series boundaries were filled by
next-observation-carried-backward or last-observation-carried-forward. The
single FDI series that was missing entirely for one city was filled with the
sample mean.

The observed-only sensitivity file was not identified from the completed
interpolated outcome column. Instead, the original pre-interpolation data were
used to form the set of observed `(City, Year)` keys, and the balanced main
panel was restricted to those keys. This removes 268 interpolated-outcome rows
while preserving the main panel's predictor definitions and values for the
remaining observations.

Interpolation was carried out before cross-validation. It does not introduce
spatial leakage, because under city-grouped cross-validation the whole time
series of a city is confined to a single fold. The observed-only sensitivity
additionally checks whether predictive performance depends on the interpolated
outcome rows themselves.

## Units

Fiscal and output aggregates (`GDP_Total`, `Gov_Expenditure`,
`Tech_Expenditure`, `FDI`) are stored in 10,000 RMB (or 10,000 USD for FDI).
Dividing by 10,000 gives 100 million RMB (100 million USD), the unit used in
Table 1 and in the axis labels of the figures. `GDP_Per_Capita` is stored in
RMB and is divided by 10,000 to obtain 10,000 RMB. `Population` is in 10,000
persons.

Winsorization is deliberately **not** applied to this file. It is part of the
modelling pipeline (`src/preprocessing.py`) and is fitted on the training
portion of every cross-validation split. Descriptive statistics for Table 1
apply the 1st/99th percentile clip at presentation time only. The outcome
`CO2_Emission` is never winsorized during model fitting; its true maximum of
457.76 Mt is preserved in the file.

## Redistribution

Carbon emission data are available from CEADs (https://www.ceads.net/) and
climate data from the National Meteorological Science Data Center
(http://data.cma.cn/). Socioeconomic indicators come from the China City
Statistical Yearbook (National Bureau of Statistics of China,
http://www.stats.gov.cn/). Please observe the terms of the original providers
when redistributing this derived panel.
