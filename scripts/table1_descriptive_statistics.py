"""Table 1 - Descriptive statistics of the analysis sample.

Reports descriptive statistics for the balanced 231-city panel after 1st/99th
percentile winsorization of the continuous variables. Winsorization is applied
here for presentation only; inside the modelling pipeline it is re-estimated
within every training fold. The outcome (CO2 emissions) is left in its original
units, so the reported maximum is a true historical record rather than a
clipped value.

Output: results/Table1_Descriptive_Statistics.csv
"""

import numpy as np
import pandas as pd

from src.config import BINARY_COLS, TO_100M_RMB, WINSOR_LOWER, WINSOR_UPPER, write_table
from src.preprocessing import load_panel

# Display labels and unit conversions. Variables are stored in their source
# units (10,000 RMB for fiscal aggregates, 10,000 persons for population) and
# are rescaled here to the units reported in the manuscript.
DISPLAY_UNITS = {
    "CO2_Emission": ("Million tonnes", 1.0),
    "HDD18": ("°C·d", 1.0),
    "CDD26": ("°C·d", 1.0),
    "Precipitation": ("mm", 1.0),
    "Built_up_Area": ("sq km", 1.0),
    "GDP_Total": ("100 Million Yuan", TO_100M_RMB),
    "GDP_Per_Capita": ("10,000 Yuan", TO_100M_RMB),
    "Sec_Industry_Ratio": ("%", 1.0),
    "Ter_Industry_Ratio": ("%", 1.0),
    "Population": ("10,000 persons", 1.0),
    "FDI": ("100 Million USD", TO_100M_RMB),
    "Gov_Expenditure": ("100 Million Yuan", TO_100M_RMB),
    "Tech_Expenditure": ("100 Million Yuan", TO_100M_RMB),
    "Buses": ("Units", 1.0),
    "Green_Coverage": ("%", 1.0),
    "Urbanization_Rate": ("%", 1.0),
    "Ind_Electricity_Ratio": ("%", 1.0),
}

IDENTIFIERS = ["City", "Year"]


def winsorized_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Clip continuous variables to their 1st and 99th percentiles."""
    continuous = [c for c in df.columns if c not in IDENTIFIERS + BINARY_COLS]
    out = df.copy()
    for col in continuous:
        series = out[col]
        out[col] = np.clip(
            series, series.quantile(WINSOR_LOWER), series.quantile(WINSOR_UPPER)
        )
    return out


def describe(series: pd.Series, is_binary: bool) -> dict:
    if is_binary:
        return {
            "Obs": int(series.count()),
            "Mean": f"{series.mean():.2f}",
            "Std. Dev.": f"{series.std():.2f}",
            "Min": "0",
            "Max": "1",
        }
    return {
        "Obs": int(series.count()),
        "Mean": f"{series.mean():.3f}",
        "Std. Dev.": f"{series.std():.3f}",
        "Min": f"{series.min():.3f}",
        "Max": f"{series.max():.3f}",
    }


def main() -> None:
    panel = winsorized_panel(load_panel())

    rows = []
    for var in panel.columns:
        if var in IDENTIFIERS:
            continue
        series = panel[var]
        if var in BINARY_COLS:
            label = var
            is_binary = True
            stats = describe(series, is_binary)
        else:
            unit, divisor = DISPLAY_UNITS.get(var, ("", 1.0))
            label = f"{var} ({unit})" if unit else var
            is_binary = False
            stats = describe(series / divisor, is_binary)
        rows.append({"Variable": label, **stats})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    write_table(table, "Table1_Descriptive_Statistics")


if __name__ == "__main__":
    main()
