from __future__ import annotations

import numpy as np
import pandas as pd


def performance_stats(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    values = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if values.empty:
        return pd.Series(dtype=float)
    nav = (1.0 + values).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    years = len(values) / periods_per_year
    annual_return = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and nav.iloc[-1] > 0 else np.nan
    annualized_mean = values.mean() * periods_per_year
    annual_vol = values.std(ddof=1) * np.sqrt(periods_per_year)
    period_rf = risk_free_rate / periods_per_year
    excess = values - period_rf
    sharpe = excess.mean() / values.std(ddof=1) * np.sqrt(periods_per_year) if values.std(ddof=1) > 0 else np.nan
    t_stat = values.mean() / (values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 and values.std(ddof=1) > 0 else np.nan
    return pd.Series(
        {
            "periods": len(values),
            "annual_return": annual_return,
            "annualized_mean_return": annualized_mean,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": drawdown.min(),
            "period_win_rate": values.gt(0).mean(),
            "period_mean": values.mean(),
            "period_t_stat": t_stat,
        }
    )


def monthly_ic(
    frame: pd.DataFrame,
    signal_col: str,
    return_col: str = "target_1m_ret",
    min_observations: int = 20,
) -> pd.DataFrame:
    rows = []
    for date, month in frame.groupby("signal_date", sort=True):
        signal = pd.to_numeric(month[signal_col], errors="coerce")
        realized = pd.to_numeric(month[return_col], errors="coerce")
        valid = signal.notna() & realized.notna()
        if valid.sum() < min_observations:
            continue
        rows.append(
            {
                "signal_date": pd.Timestamp(date),
                "ic": signal[valid].corr(realized[valid], method="pearson"),
                "rank_ic": signal[valid].corr(realized[valid], method="spearman"),
                "n": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows, columns=["signal_date", "ic", "rank_ic", "n"])


def ic_summary(ic: pd.DataFrame) -> pd.Series:
    result = {}
    for source, prefix in [("ic", "ic"), ("rank_ic", "rank_ic")]:
        values = pd.to_numeric(ic[source], errors="coerce").dropna()
        mean = values.mean()
        std = values.std(ddof=1)
        result[prefix] = mean
        result[f"{prefix}ir"] = mean / std if std > 0 else np.nan
        result[f"{prefix}_t_stat"] = mean / (std / np.sqrt(len(values))) if len(values) > 1 and std > 0 else np.nan
        result[f"{prefix}_positive_rate"] = values.gt(0).mean() if len(values) else np.nan
    result["months"] = len(ic)
    return pd.Series(result)


def annual_performance(returns: pd.Series, periods_per_year: int = 252) -> pd.DataFrame:
    rows = []
    values = returns.dropna()
    for year, group in values.groupby(pd.to_datetime(values.index).year):
        rows.append({"year": int(year), **performance_stats(group, periods_per_year).to_dict()})
    return pd.DataFrame(rows)
