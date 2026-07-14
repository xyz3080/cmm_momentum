from __future__ import annotations

import numpy as np
import pandas as pd


BARRA_STYLE_COLS = [
    "style_size",
    "style_liquidity",
    "style_momentum",
    "style_volatility",
    "style_profitability",
    "style_growth",
    "style_leverage",
]


def cs_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def _mean_existing_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return pd.Series(0.0, index=frame.index)
    return frame[existing].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.0)


def add_barra_style_exposures(
    frame: pd.DataFrame,
    return_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    if return_cols is None:
        return_cols = sorted(
            [column for column in out.columns if column.startswith("ret_lag_")],
            key=lambda column: int(column.rsplit("_", 1)[-1]),
            reverse=True,
        )
    return_cols = [column for column in return_cols if column in out.columns]
    size_col = "z_nmv" if "z_nmv" in out.columns else "z_tmv"
    out["style_size"] = pd.to_numeric(out.get(size_col, 0.0), errors="coerce").fillna(0.0)
    out["style_liquidity"] = _mean_existing_columns(
        out,
        ["z_amount", "z_volume", "z_pv_turnover_1m", "z_pv_turnover_3m", "z_pv_amount_mean_1m", "z_pv_amount_mean_3m"],
    )
    if return_cols:
        returns = out[return_cols].apply(pd.to_numeric, errors="coerce")
        out["style_momentum"] = returns.sum(axis=1).fillna(0.0)
        out["style_volatility"] = returns.std(axis=1, ddof=0).fillna(0.0)
    else:
        out["style_momentum"] = 0.0
        out["style_volatility"] = 0.0
    out["style_profitability"] = _mean_existing_columns(
        out,
        [
            "z_rel_net_profit_ttm_to_total_assets",
            "z_rel_net_profit_ttm_to_equity",
            "z_rel_net_profit_ttm_to_operating_revenue_ttm",
            "z_rel_gross_profit_ttm_to_operating_revenue_ttm",
        ],
    )
    out["style_growth"] = _mean_existing_columns(
        out,
        ["z_oper_revenue_gr_ttm", "z_net_profit_gr_ttm", "z_eps_gr_ttm", "z_net_asset_gr"],
    )
    out["style_leverage"] = _mean_existing_columns(
        out,
        ["z_rel_total_liability_to_total_assets", "z_rel_total_liability_to_equity", "z_rel_total_assets_to_equity"],
    )
    for column in BARRA_STYLE_COLS:
        out[column] = out.groupby("signal_date", group_keys=False)[column].transform(cs_zscore).fillna(0.0)
    return out, BARRA_STYLE_COLS.copy()


def neutralize_by_size_industry(frame: pd.DataFrame, signal_col: str) -> pd.Series:
    residuals = pd.Series(index=frame.index, dtype=float)
    size_col = "z_nmv" if "z_nmv" in frame.columns else "z_tmv"
    required = {"signal_date", signal_col, size_col, "ind1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"neutralization missing columns: {sorted(missing)}")
    for _, group in frame.groupby("signal_date", sort=True):
        y = pd.to_numeric(group[signal_col], errors="coerce").astype(float)
        valid = y.notna()
        if valid.sum() < 2:
            continue
        x = pd.DataFrame({"const": 1.0, size_col: pd.to_numeric(group[size_col], errors="coerce").fillna(0.0)})
        industry = pd.get_dummies(group["ind1"].fillna("Unknown").astype(str), prefix="ind", drop_first=True, dtype=float)
        x = pd.concat([x, industry], axis=1).astype(float)
        beta, *_ = np.linalg.lstsq(x.loc[valid].to_numpy(), y.loc[valid].to_numpy(), rcond=None)
        residuals.loc[group.index] = y.to_numpy() - x.to_numpy() @ beta
    return residuals


def neutralize_by_barra_style(
    frame: pd.DataFrame,
    signal_col: str,
    style_cols: list[str] | None = None,
    industry_col: str = "ind1",
) -> pd.Series:
    style_cols = style_cols or [column for column in BARRA_STYLE_COLS if column in frame.columns]
    residuals = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("signal_date", sort=True):
        y = pd.to_numeric(group[signal_col], errors="coerce").astype(float)
        valid = y.notna()
        if valid.sum() < max(50, len(style_cols) + 2):
            continue
        styles = group.loc[valid, style_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        industry = pd.get_dummies(
            group.loc[valid, industry_col].fillna("Unknown").astype(str),
            prefix="ind",
            drop_first=True,
            dtype=float,
        )
        x = pd.concat([pd.DataFrame({"const": 1.0}, index=styles.index), styles, industry], axis=1)
        beta, *_ = np.linalg.lstsq(x.to_numpy(), y.loc[valid].to_numpy(), rcond=None)
        residuals.loc[styles.index] = y.loc[valid].to_numpy() - x.to_numpy() @ beta
    return residuals
