from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "code"))

from backtest import (  # noqa: E402
    BacktestConfig,
    DailyMarketData,
    performance_stats,
    prepare_trade_flags,
    run_monthly_factor_backtest,
)
from backtest.engine import PortfolioState, rebalance_signed, run_equal_weight_groups  # noqa: E402


def test_performance_stats_separates_cagr_from_annualized_mean() -> None:
    returns = pd.Series(
        [0.10, -0.10],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        dtype=float,
    )

    stats = performance_stats(returns, periods_per_year=252)

    expected_cagr = (1.10 * 0.90) ** (252 / 2) - 1
    assert np.isclose(stats["annual_return"], expected_cagr)
    assert np.isclose(stats["annualized_mean_return"], 0.0)


def test_missing_trade_flags_are_not_tradable() -> None:
    frame = pd.DataFrame(
        {
            "stock_id": ["a", "b"],
            "trade_date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
        }
    )
    flags = pd.DataFrame(
        {
            "stock_id": ["a"],
            "trade_date": pd.to_datetime(["2024-02-01"]),
            "limit_up": [False],
            "limit_down": [False],
            "is_tradable": [True],
        }
    )

    merged = prepare_trade_flags(frame, flags)

    assert bool(merged.loc[merged.stock_id.eq("a"), "can_buy"].iloc[0])
    assert not bool(merged.loc[merged.stock_id.eq("b"), "can_buy"].iloc[0])
    assert not bool(merged.loc[merged.stock_id.eq("b"), "can_sell"].iloc[0])


def test_daily_market_data_excludes_trade_date_and_includes_exit_date() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-05"]),
            "stock_id": ["a", "a", "a"],
            "ret": [0.01, 0.02, 0.03],
        }
    )
    market = DailyMarketData.from_frame(daily)

    assert market.period_dates(pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-05")) == (
        pd.Timestamp("2024-02-02"),
        pd.Timestamp("2024-02-05"),
    )


def test_monthly_result_uses_equal_weight_groups_and_full_section_long_short() -> None:
    stocks = ["a", "b", "c", "d"]
    frame = pd.DataFrame(
        {
            "stock_id": stocks,
            "signal_date": pd.Timestamp("2024-01-31"),
            "trade_date": pd.Timestamp("2024-02-01"),
            "exit_date": pd.Timestamp("2024-02-05"),
            "signal": [-2.0, -1.0, 1.0, 2.0],
            "target_1m_ret": [-0.02, -0.01, 0.01, 0.02],
            "limit_up": False,
            "limit_down": False,
            "is_tradable": True,
            "can_buy": True,
            "can_sell": True,
        }
    )
    daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-02-02")] * 4 + [pd.Timestamp("2024-02-05")] * 4,
            "stock_id": stocks * 2,
            "ret": [-0.01, -0.005, 0.005, 0.01] * 2,
        }
    )
    config = BacktestConfig(n_groups=2, min_ic_observations=2, periods_per_year=252)

    result = run_monthly_factor_backtest(frame, "signal", DailyMarketData.from_frame(daily), config)

    assert list(result.group_returns.columns) == [1, 2]
    assert result.long_short_returns.name == "signal_long_short"
    assert not result.long_returns.empty
    assert not result.short_returns.empty
    assert set(result.performance.index) == {"long", "short", "long_short"}
    assert {"ic", "rank_ic", "n"}.issubset(result.monthly_ic.columns)
    assert result.turnover["long_short_held_stock_observations"].sum() == 8
    assert result.turnover["long_short_missing_returns"].sum() == 0


def test_backtest_config_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="n_groups"):
        BacktestConfig(n_groups=1)
    with pytest.raises(ValueError, match="periods_per_year"):
        BacktestConfig(periods_per_year=0)


def test_equal_weight_transaction_cost_is_charged_once() -> None:
    frame = pd.DataFrame(
        {
            "stock_id": ["a", "b"],
            "signal_date": pd.Timestamp("2024-01-31"),
            "trade_date": pd.Timestamp("2024-02-01"),
            "exit_date": pd.Timestamp("2024-02-02"),
            "signal": [-1.0, 1.0],
            "can_buy": True,
            "can_sell": True,
        }
    )
    daily = DailyMarketData.from_frame(
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-02-02")] * 2,
                "stock_id": ["a", "b"],
                "ret": [0.0, 0.0],
            }
        )
    )
    config = BacktestConfig(n_groups=2, transaction_cost_bps=100.0, min_ic_observations=2)

    group_returns, _ = run_equal_weight_groups(frame, "signal", daily, config)

    assert np.isclose(group_returns.iloc[0, 0], -0.01)
    assert np.isclose(group_returns.iloc[0, 1], -0.01)


def test_limit_constraints_keep_existing_signed_position() -> None:
    current = {"a": 0.5, "b": -0.5}
    target = {"a": 0.0, "b": 0.0}

    weights, turnover, _ = rebalance_signed(
        current,
        target,
        can_buy={"a": True, "b": False},
        can_sell={"a": False, "b": True},
        transaction_cost_bps=0.0,
    )

    assert weights == current
    assert turnover == 0.0
