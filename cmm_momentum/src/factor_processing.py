from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(series: pd.Series, limits: tuple[float, float]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    lower, upper = limits
    if not (0 <= lower < upper <= 1):
        raise ValueError("winsorize limits must satisfy 0 <= lower < upper <= 1")
    if values.notna().sum() == 0:
        return values
    return values.clip(values.quantile(lower), values.quantile(upper))


def apply_monthly_winsorize(
    df: pd.DataFrame,
    signal_col: str,
    limits: tuple[float, float],
    date_col: str = "signal_date",
) -> pd.Series:
    return df.groupby(date_col, group_keys=False)[signal_col].transform(
        lambda s: winsorize_series(s.replace([np.inf, -np.inf], np.nan), limits)
    )
