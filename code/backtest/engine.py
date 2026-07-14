from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import BacktestConfig
from .data import DailyMarketData
from .weights import assign_groups, build_equal_weights, build_signed_weights, split_signed_weights


@dataclass
class PortfolioState:
    weights: dict[str, float]
    cash: float = 0.0


def rebalance_signed(
    current: dict[str, float],
    target: dict[str, float],
    can_buy: dict[str, bool],
    can_sell: dict[str, bool],
    transaction_cost_bps: float,
) -> tuple[dict[str, float], float, float]:
    weights = dict(current)
    turnover = 0.0
    for stock_id in set(weights) | set(target):
        existing = weights.get(stock_id, 0.0)
        desired = target.get(stock_id, 0.0)
        delta = desired - existing
        if abs(delta) <= 1e-12:
            continue
        if delta > 0 and not can_buy.get(stock_id, False):
            continue
        if delta < 0 and not can_sell.get(stock_id, False):
            continue
        weights[stock_id] = desired
        turnover += abs(delta)
    weights = {name: value for name, value in weights.items() if abs(value) > 1e-12}
    return weights, turnover, turnover * transaction_cost_bps / 10_000.0


def rebalance_long_only(
    state: PortfolioState,
    target: dict[str, float],
    can_buy: dict[str, bool],
    can_sell: dict[str, bool],
    transaction_cost_bps: float,
) -> tuple[PortfolioState, float, float]:
    holdings = dict(state.weights)
    cash = state.cash
    turnover = 0.0
    stock_set = set(holdings) | set(target)
    for stock_id in stock_set:
        current = holdings.get(stock_id, 0.0)
        desired = target.get(stock_id, 0.0)
        if desired < current and can_sell.get(stock_id, False):
            sold = current - desired
            holdings[stock_id] = desired
            cash += sold
            turnover += sold
    gaps = {
        stock_id: target.get(stock_id, 0.0) - holdings.get(stock_id, 0.0)
        for stock_id in stock_set
        if target.get(stock_id, 0.0) > holdings.get(stock_id, 0.0) and can_buy.get(stock_id, False)
    }
    total_gap = sum(gaps.values())
    scale = min(1.0, cash / total_gap) if total_gap > 0 else 0.0
    for stock_id, gap in gaps.items():
        bought = gap * scale
        holdings[stock_id] = holdings.get(stock_id, 0.0) + bought
        cash -= bought
        turnover += bought
    cost = turnover * transaction_cost_bps / 10_000.0
    cash -= cost
    holdings = {name: value for name, value in holdings.items() if value > 1e-12}
    return PortfolioState(holdings, cash), turnover, cost


def advance_signed_one_day(
    weights: dict[str, float],
    stock_returns: dict[str, float],
    cost: float = 0.0,
) -> tuple[float, dict[str, float], int]:
    missing = sum(name not in stock_returns for name in weights)
    pnl = sum(weight * float(stock_returns.get(name, 0.0)) for name, weight in weights.items())
    daily_return = pnl - cost
    end_value = 1.0 + daily_return
    if end_value <= 0:
        return -1.0, {}, missing
    next_weights = {
        name: weight * (1.0 + float(stock_returns.get(name, 0.0))) / end_value
        for name, weight in weights.items()
    }
    return daily_return, {name: value for name, value in next_weights.items() if abs(value) > 1e-12}, missing


def advance_long_only_one_day(
    state: PortfolioState,
    stock_returns: dict[str, float],
) -> tuple[float, PortfolioState, int]:
    missing = sum(name not in stock_returns for name in state.weights)
    values = {
        name: weight * (1.0 + float(stock_returns.get(name, 0.0)))
        for name, weight in state.weights.items()
    }
    end_value = state.cash + sum(values.values())
    if end_value <= 0:
        return -1.0, PortfolioState({}, 0.0), missing
    next_state = PortfolioState(
        {name: value / end_value for name, value in values.items() if value > 1e-12},
        state.cash / end_value,
    )
    return end_value - 1.0, next_state, missing


def run_equal_weight_groups(
    frame: pd.DataFrame,
    signal_col: str,
    market: DailyMarketData,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = {group: PortfolioState({}, 1.0) for group in range(1, config.n_groups + 1)}
    daily_parts = []
    diagnostics = []
    for signal_date, raw_month in frame.groupby("signal_date", sort=True):
        month = raw_month.dropna(subset=[signal_col]).copy()
        month["group"] = assign_groups(month, signal_col, config.n_groups)
        if month["group"].isna().all():
            continue
        can_buy = month.set_index("stock_id")["can_buy"].astype(bool).to_dict()
        can_sell = month.set_index("stock_id")["can_sell"].astype(bool).to_dict()
        trade_date = pd.Timestamp(month["trade_date"].iloc[0])
        exit_date = pd.Timestamp(month["exit_date"].iloc[0])
        period = {}
        row = {"signal_date": pd.Timestamp(signal_date)}
        for group in range(1, config.n_groups + 1):
            target = build_equal_weights(month, group)
            state, turnover, cost = rebalance_long_only(
                states[group], target, can_buy, can_sell, config.transaction_cost_bps
            )
            values = []
            missing = 0
            held_stock_observations = 0
            for date in market.period_dates(trade_date, exit_date):
                held_stock_observations += len(state.weights)
                daily_return, state, count = advance_long_only_one_day(state, market.returns_by_date.get(date, {}))
                values.append((date, daily_return))
                missing += count
            states[group] = state
            period[group] = pd.Series(dict(values), dtype=float)
            row[f"group_{group}_turnover"] = turnover
            row[f"group_{group}_missing_returns"] = missing
            row[f"group_{group}_held_stock_observations"] = held_stock_observations
        daily_parts.append(pd.DataFrame(period))
        diagnostics.append(row)
    result = pd.concat(daily_parts).sort_index() if daily_parts else pd.DataFrame(columns=range(1, config.n_groups + 1))
    result.index.name = "date"
    return result, pd.DataFrame(diagnostics)


def run_signed_portfolio_with_legs(
    frame: pd.DataFrame,
    signal_col: str,
    market: DailyMarketData,
    config: BacktestConfig,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    states = {"long_short": {}, "long": {}, "short": {}}
    rows = {name: [] for name in states}
    diagnostics = []
    for signal_date, month in frame.groupby("signal_date", sort=True):
        target = build_signed_weights(month, signal_col, config.gross, config.max_abs_weight)
        target_long, target_short = split_signed_weights(target)
        targets = {"long_short": target, "long": target_long, "short": target_short}
        can_buy = month.set_index("stock_id")["can_buy"].astype(bool).to_dict()
        can_sell = month.set_index("stock_id")["can_sell"].astype(bool).to_dict()
        trade_date = pd.Timestamp(month["trade_date"].iloc[0])
        exit_date = pd.Timestamp(month["exit_date"].iloc[0])
        diag = {"signal_date": pd.Timestamp(signal_date)}
        for name, target_weights in targets.items():
            states[name], turnover, cost = rebalance_signed(
                states[name], target_weights, can_buy, can_sell, config.transaction_cost_bps
            )
            pending_cost = cost
            missing = 0
            held_stock_observations = 0
            for date in market.period_dates(trade_date, exit_date):
                held_stock_observations += len(states[name])
                daily_return, states[name], count = advance_signed_one_day(
                    states[name], market.returns_by_date.get(date, {}), pending_cost
                )
                pending_cost = 0.0
                missing += count
                rows[name].append({"date": date, "return": daily_return})
            diag[f"{name}_turnover"] = turnover
            diag[f"{name}_cost"] = cost
            diag[f"{name}_missing_returns"] = missing
            diag[f"{name}_held_stock_observations"] = held_stock_observations
            diag[f"{name}_gross_exposure"] = sum(abs(value) for value in states[name].values())
            diag[f"{name}_net_exposure"] = sum(states[name].values())
            diag[f"{name}_held_names"] = len(states[name])
        diagnostics.append(diag)

    def as_series(name: str) -> pd.Series:
        result = pd.Series({row["date"]: row["return"] for row in rows[name]}, name=f"{signal_col}_{name}")
        result.index.name = "date"
        return result.sort_index()

    return as_series("long_short"), as_series("long"), as_series("short"), pd.DataFrame(diagnostics)
