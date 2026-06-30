from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DAILY_DIR = WORKSPACE_ROOT / "data" / "daily"


def normalize_daily_symbol(symbol: pd.Series) -> pd.Series:
    s = symbol.astype(str).str.strip()
    exchange = s.str[:2]
    code = s.str[2:]
    suffix = exchange.map({"SH": ".SH", "SZ": ".SZ", "BJ": ".BJ"}).fillna("")
    return code + suffix


def limit_rate(stock_id: pd.Series, trade_date: pd.Series, st: pd.Series | None = None) -> pd.Series:
    stock = stock_id.astype(str)
    date = pd.to_datetime(trade_date)
    rate = pd.Series(0.10, index=stock.index)
    rate[stock.str.startswith(("688", "300")) & (date >= pd.Timestamp("2020-08-24"))] = 0.20
    rate[stock.str.startswith(("8", "4"))] = 0.30
    if st is not None:
        rate[pd.to_numeric(st, errors="coerce").fillna(0).astype(int).eq(1)] = 0.05
    return rate


def load_trade_flags(trade_dates: pd.Series, daily_dir: Path = DAILY_DIR) -> pd.DataFrame:
    pieces = []
    for d in sorted(pd.to_datetime(trade_dates.dropna().unique())):
        path = daily_dir / f"{d:%Y-%m-%d}.csv"
        if not path.exists():
            continue
        daily = pd.read_csv(path, usecols=["date", "symbol", "close", "preClose", "ret", "st"])
        daily["trade_date"] = pd.to_datetime(daily["date"])
        daily["stock_id"] = normalize_daily_symbol(daily["symbol"])
        rate = limit_rate(daily["stock_id"], daily["trade_date"], daily["st"])
        ret = pd.to_numeric(daily["ret"], errors="coerce")
        close = pd.to_numeric(daily["close"], errors="coerce")
        pre_close = pd.to_numeric(daily["preClose"], errors="coerce")
        daily["limit_up"] = (ret >= rate - 0.002) | (close >= pre_close * (1 + rate) * 0.998)
        daily["limit_down"] = (ret <= -rate + 0.002) | (close <= pre_close * (1 - rate) * 1.002)
        pieces.append(daily[["stock_id", "trade_date", "limit_up", "limit_down"]])
    if not pieces:
        return pd.DataFrame(columns=["stock_id", "trade_date", "limit_up", "limit_down"])
    return pd.concat(pieces, ignore_index=True).drop_duplicates(["stock_id", "trade_date"])


def load_signal_market_caps(signal_dates: pd.Series, daily_dir: Path = DAILY_DIR) -> pd.DataFrame:
    pieces = []
    for d in sorted(pd.to_datetime(signal_dates.dropna().unique())):
        path = daily_dir / f"{d:%Y-%m-%d}.csv"
        if not path.exists():
            continue
        daily = pd.read_csv(path, usecols=["date", "symbol", "tmv"])
        daily["signal_date"] = pd.to_datetime(daily["date"])
        daily["stock_id"] = normalize_daily_symbol(daily["symbol"])
        daily["market_cap"] = pd.to_numeric(daily["tmv"], errors="coerce")
        pieces.append(daily[["stock_id", "signal_date", "market_cap"]])
    if not pieces:
        return pd.DataFrame(columns=["stock_id", "signal_date", "market_cap"])
    return pd.concat(pieces, ignore_index=True).drop_duplicates(["stock_id", "signal_date"])


def cs_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def neutralize_by_size_industry(df: pd.DataFrame, signal_col: str) -> pd.Series:
    residuals = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("signal_date", sort=True):
        y = pd.to_numeric(g[signal_col], errors="coerce").astype(float)
        x = pd.DataFrame({"const": 1.0, "z_tmv": pd.to_numeric(g["z_tmv"], errors="coerce").fillna(0.0)})
        industry = pd.get_dummies(g["ind1"].fillna("Unknown").astype(str), prefix="ind", drop_first=True, dtype=float)
        x = pd.concat([x, industry], axis=1).astype(float)
        valid = y.notna()
        beta, *_ = np.linalg.lstsq(x.loc[valid].to_numpy(), y.loc[valid].to_numpy(), rcond=None)
        fitted = x.to_numpy() @ beta
        residuals.loc[g.index] = y.to_numpy() - fitted
    return residuals


def build_target_weights(month: pd.DataFrame, signal_col: str, decile: int, weighting: str = "equal") -> dict[str, float]:
    target = month.loc[month["decile"].eq(decile)].copy()
    if target.empty:
        return {}
    if weighting == "equal":
        return {stock_id: 1.0 / len(target) for stock_id in target["stock_id"]}
    if weighting == "value":
        cap = pd.to_numeric(target["market_cap"], errors="coerce").clip(lower=0)
        if cap.notna().sum() == 0 or cap.sum() <= 0:
            return {stock_id: 1.0 / len(target) for stock_id in target["stock_id"]}
        weights = cap / cap.sum()
        return dict(zip(target["stock_id"], weights))
    raise ValueError(f"Unknown weighting: {weighting}")


def monthly_rebalance_return(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    stock_returns: dict[str, float],
    can_buy: dict[str, bool],
    can_sell: dict[str, bool],
    cash: float = 0.0,
) -> tuple[float, dict[str, float], float]:
    holdings = dict(current_weights)
    stock_set = set(holdings) | set(target_weights)

    for stock_id in stock_set:
        current = holdings.get(stock_id, 0.0)
        target = target_weights.get(stock_id, 0.0)
        if target < current and can_sell.get(stock_id, True):
            cash += current - target
            holdings[stock_id] = target

    total_gap = 0.0
    for stock_id in stock_set:
        current = holdings.get(stock_id, 0.0)
        target = target_weights.get(stock_id, 0.0)
        if target > current and can_buy.get(stock_id, True):
            total_gap += target - current
    if total_gap > 0:
        scale = min(1.0, cash / total_gap)
        for stock_id in stock_set:
            current = holdings.get(stock_id, 0.0)
            target = target_weights.get(stock_id, 0.0)
            if target > current and can_buy.get(stock_id, True):
                buy = (target - current) * scale
                holdings[stock_id] = current + buy
                cash -= buy

    realized = {stock_id: holdings.get(stock_id, 0.0) * (1.0 + float(stock_returns.get(stock_id, 0.0))) for stock_id in holdings}
    end_value = cash + sum(realized.values())
    if end_value <= 0:
        return -1.0, {}, 0.0
    next_weights = {stock_id: value / end_value for stock_id, value in realized.items() if value > 1e-12}
    next_cash = cash / end_value
    return end_value - 1.0, next_weights, next_cash


def backtest_deciles(df: pd.DataFrame, signal_col: str, weighting: str = "equal", n_deciles: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    previous_weights = {decile: {} for decile in range(1, n_deciles + 1)}
    previous_cash = {decile: 1.0 for decile in range(1, n_deciles + 1)}

    for date, month in df.groupby("signal_date", sort=True):
        month = month.copy()
        if month[signal_col].nunique() < n_deciles:
            continue
        month["decile"] = pd.qcut(month[signal_col].rank(method="first"), n_deciles, labels=False) + 1
        stock_returns = month.set_index("stock_id")["target_1m_ret"].to_dict()
        can_buy = (~month.set_index("stock_id")["limit_up"]).to_dict()
        can_sell = (~month.set_index("stock_id")["limit_down"]).to_dict()
        row = {"signal_date": date}
        for decile in range(1, n_deciles + 1):
            target_weights = build_target_weights(month, signal_col, decile, weighting)
            ret, next_weights, next_cash = monthly_rebalance_return(
                previous_weights[decile],
                target_weights,
                stock_returns,
                can_buy,
                can_sell,
                previous_cash[decile],
            )
            previous_weights[decile] = next_weights
            previous_cash[decile] = next_cash
            row[decile] = ret
        rows.append(row)

    decile_returns = pd.DataFrame(rows).set_index("signal_date").sort_index()
    long_short = decile_returns[n_deciles] - decile_returns[1]
    return decile_returns, long_short


def perf_stats(monthly_returns: pd.Series) -> pd.Series:
    monthly_returns = monthly_returns.dropna()
    nav = (1.0 + monthly_returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    ann_ret = monthly_returns.mean() * 12
    ann_vol = monthly_returns.std(ddof=1) * np.sqrt(12)
    t_stat = (
        monthly_returns.mean() / (monthly_returns.std(ddof=1) / np.sqrt(len(monthly_returns)))
        if len(monthly_returns) > 1 and monthly_returns.std(ddof=1) > 0
        else np.nan
    )
    return pd.Series(
        {
            "months": len(monthly_returns),
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "max_drawdown": drawdown.min() if len(drawdown) else np.nan,
            "monthly_win_rate": (monthly_returns > 0).mean() if len(monthly_returns) else np.nan,
            "monthly_mean": monthly_returns.mean() if len(monthly_returns) else np.nan,
            "monthly_t_stat": t_stat,
        }
    )

