from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_gross(weights: pd.Series, gross: float = 1.0) -> pd.Series:
    exposure = weights.abs().sum()
    if not np.isfinite(exposure) or exposure <= 0:
        return pd.Series(0.0, index=weights.index)
    return weights / exposure * gross


def truncate_weights(weights: pd.Series, max_abs_weight: float, gross: float = 1.0) -> pd.Series:
    if max_abs_weight <= 0:
        return normalize_gross(weights, gross)
    capped = normalize_gross(weights, gross)
    for _ in range(100):
        over = capped.abs().gt(max_abs_weight + 1e-12)
        if not over.any():
            break
        capped.loc[over] = np.sign(capped.loc[over]) * max_abs_weight
        free = ~over
        remaining = gross - capped.loc[over].abs().sum()
        if remaining <= 1e-12 or not free.any() or capped.loc[free].abs().sum() <= 1e-12:
            break
        capped.loc[free] = normalize_gross(capped.loc[free], remaining)
    return capped


def build_signed_weights(
    month: pd.DataFrame,
    signal_col: str,
    gross: float = 1.0,
    max_abs_weight: float = 0.0,
    group_col: str | None = None,
    demean: bool = True,
) -> dict[str, float]:
    columns = ["stock_id", signal_col] + ([group_col] if group_col else [])
    sample = month[columns].dropna(subset=["stock_id", signal_col]).copy()
    signal = pd.to_numeric(sample[signal_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = signal.notna()
    sample = sample.loc[valid]
    signal = signal.loc[valid].astype(float)
    if sample.empty:
        return {}
    if group_col:
        groups = sample[group_col].fillna("Unknown").astype(str)
        signal = signal - signal.groupby(groups).transform("mean")
    if demean:
        signal = signal - signal.mean()
    weights = truncate_weights(signal, max_abs_weight, gross)
    weights.index = sample["stock_id"].astype(str).to_numpy()
    return weights[weights.abs().gt(1e-12)].to_dict()


def build_worldquant_weights(
    month: pd.DataFrame,
    signal_col: str,
    group_col: str | None = None,
    gross: float = 1.0,
    max_abs_weight: float = 0.0,
    demean: bool = True,
) -> dict[str, float]:
    return build_signed_weights(month, signal_col, gross, max_abs_weight, group_col, demean)


def split_signed_weights(weights: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    series = pd.Series(weights, dtype=float)
    long = normalize_gross(series[series.gt(0)], 1.0)
    short = normalize_gross(series[series.lt(0)], 1.0)
    return long.to_dict(), short.to_dict()


def assign_groups(month: pd.DataFrame, signal_col: str, n_groups: int) -> pd.Series:
    signal = pd.to_numeric(month[signal_col], errors="coerce")
    valid = signal.notna()
    groups = pd.Series(pd.NA, index=month.index, dtype="Int64")
    if valid.sum() >= n_groups and signal.loc[valid].nunique() >= n_groups:
        groups.loc[valid] = pd.qcut(signal.loc[valid].rank(method="first"), n_groups, labels=False).astype(int) + 1
    return groups


def build_equal_weights(month: pd.DataFrame, group: int) -> dict[str, float]:
    selected = month.loc[month["group"].eq(group), "stock_id"].dropna().astype(str).drop_duplicates()
    if selected.empty:
        return {}
    return dict.fromkeys(selected, 1.0 / len(selected))
