from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import BacktestConfig
from .data import DailyMarketData
from .engine import run_equal_weight_groups, run_signed_portfolio_with_legs
from .metrics import annual_performance, ic_summary, monthly_ic, performance_stats
from .plots import plot_group_and_long_short_nav


@dataclass
class BacktestResult:
    signal_col: str
    group_returns: pd.DataFrame
    long_short_returns: pd.Series
    long_returns: pd.Series
    short_returns: pd.Series
    performance: pd.DataFrame
    annual_performance: dict[str, pd.DataFrame]
    monthly_ic: pd.DataFrame
    ic_summary: pd.Series
    turnover: pd.DataFrame
    group_diagnostics: pd.DataFrame


def run_monthly_factor_backtest(
    frame: pd.DataFrame,
    signal_col: str,
    market: DailyMarketData,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    required = {"stock_id", "signal_date", "trade_date", "exit_date", signal_col, "target_1m_ret"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"factor frame missing columns: {sorted(missing)}")
    prepared = frame.copy()
    if "can_buy" not in prepared or "can_sell" not in prepared:
        tradable = prepared.get("is_tradable", pd.Series(True, index=prepared.index)).eq(True)
        prepared["can_buy"] = tradable & ~prepared.get("limit_up", pd.Series(False, index=prepared.index)).eq(True)
        prepared["can_sell"] = tradable & ~prepared.get("limit_down", pd.Series(False, index=prepared.index)).eq(True)
    groups, group_diagnostics = run_equal_weight_groups(prepared, signal_col, market, config)
    long_short, long, short, turnover = run_signed_portfolio_with_legs(prepared, signal_col, market, config)
    legs = {"long": long, "short": short, "long_short": long_short}
    performance = pd.DataFrame(
        {name: performance_stats(values, config.periods_per_year, config.risk_free_rate) for name, values in legs.items()}
    ).T
    annual = {name: annual_performance(values, config.periods_per_year) for name, values in legs.items()}
    ic = monthly_ic(prepared, signal_col, min_observations=config.min_ic_observations)
    return BacktestResult(
        signal_col=signal_col,
        group_returns=groups,
        long_short_returns=long_short,
        long_returns=long,
        short_returns=short,
        performance=performance,
        annual_performance=annual,
        monthly_ic=ic,
        ic_summary=ic_summary(ic),
        turnover=turnover,
        group_diagnostics=group_diagnostics,
    )


def write_backtest_tables(result: BacktestResult, output_dir: Path, prefix: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "performance": output_dir / f"{prefix}_performance.csv",
        "ic": output_dir / f"{prefix}_monthly_ic.csv",
        "turnover": output_dir / f"{prefix}_turnover.csv",
    }
    result.performance.to_csv(paths["performance"], index_label="portfolio")
    result.monthly_ic.to_csv(paths["ic"], index=False)
    result.turnover.to_csv(paths["turnover"], index=False)
    return paths


def write_backtest_outputs(
    result: BacktestResult,
    output_dir: Path,
    prefix: str,
    title: str,
) -> dict[str, Path]:
    paths = write_backtest_tables(result, output_dir, prefix)
    paths["nav_plot"] = plot_group_and_long_short_nav(
        result.group_returns,
        result.long_short_returns,
        output_dir / f"{prefix}_group_long_short_nav.png",
        title,
    )
    return paths
