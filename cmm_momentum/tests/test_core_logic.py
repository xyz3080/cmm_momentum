from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in [PROJECT_ROOT / "src", WORKSPACE_ROOT / "code"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_clean_pipeline import (  # noqa: E402
    FINANCIAL_SOURCE_CANDIDATES,
    MODEL_ANNOUNCEMENT_FEATURES,
    MODEL_DAILY_FEATURES,
    MODEL_FINANCIAL_FEATURES,
    RELATIVE_FUNDAMENTAL_PAIRS,
    attach_point_in_time_financials,
    build_signal_calendar,
)
from factor_processing import apply_monthly_winsorize  # noqa: E402
from backtest import (  # noqa: E402
    add_barra_style_exposures,
    build_worldquant_weights,
    split_signed_weights,
)
from style_exposure_workflow import estimate_factor_style_exposure  # noqa: E402
from model_compare_workflow import factor_distribution_tables  # noqa: E402
from model_compare_workflow import prepare_factor_signals  # noqa: E402


def test_signal_calendar_uses_next_trading_day_entry() -> None:
    dates = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])
    calendar = build_signal_calendar(pd.Series(dates), entry_lag_days=1)
    assert calendar.loc[0, "signal_date"] == pd.Timestamp("2024-01-31")
    assert calendar.loc[0, "trade_date"] == pd.Timestamp("2024-02-01")
    assert calendar.loc[0, "exit_date"] == pd.Timestamp("2024-03-01")


def test_financial_announcement_date_becomes_point_in_time_recency_feature() -> None:
    samples = pd.DataFrame(
        {
            "stock_id": ["a", "a"],
            "signal_date": pd.to_datetime(["2024-04-30", "2024-05-31"]),
        }
    )
    financial = pd.DataFrame(
        {
            "stock_id": ["a", "a"],
            "report_date": pd.to_datetime(["2023-12-31", "2024-03-31"]),
            "public_date": pd.to_datetime(["2024-04-20", "2024-05-15"]),
            "net_profit": [1.0, 2.0],
        }
    )

    merged = attach_point_in_time_financials(samples, financial, ["net_profit"])

    assert merged["financial_public_date"].tolist() == [
        pd.Timestamp("2024-04-20"),
        pd.Timestamp("2024-05-15"),
    ]
    assert merged["financial_announcement_recency_days"].tolist() == [10.0, 16.0]
    assert (merged["financial_public_date"] <= merged["signal_date"]).all()


def test_monthly_winsorize_is_cross_sectional() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-31")] * 5 + [pd.Timestamp("2024-02-29")] * 5,
            "factor": [0, 1, 2, 3, 100, -100, 1, 2, 3, 4],
        }
    )
    clipped = apply_monthly_winsorize(frame, "factor", (0.2, 0.8))
    assert clipped.iloc[:5].min() >= np.quantile([0, 1, 2, 3, 100], 0.2)
    assert clipped.iloc[:5].max() <= np.quantile([0, 1, 2, 3, 100], 0.8)
    assert clipped.iloc[5:].min() >= np.quantile([-100, 1, 2, 3, 4], 0.2)
    assert clipped.iloc[5:].max() <= np.quantile([-100, 1, 2, 3, 4], 0.8)


def test_worldquant_weights_are_demeaned_and_gross_normalized() -> None:
    month = pd.DataFrame({"stock_id": ["a", "b", "c"], "signal": [1.0, 2.0, 4.0]})
    weights = pd.Series(build_worldquant_weights(month, "signal", demean=True))
    assert abs(weights.sum()) < 1e-12
    assert abs(weights.abs().sum() - 1.0) < 1e-12


def test_full_cross_section_legs_preserve_names_and_relative_weights() -> None:
    signed = {"a": 0.3, "b": 0.2, "c": -0.1, "d": -0.4}
    long_leg, short_leg = split_signed_weights(signed)

    assert set(long_leg) == {"a", "b"}
    assert set(short_leg) == {"c", "d"}
    assert np.isclose(sum(long_leg.values()), 1.0)
    assert np.isclose(sum(abs(value) for value in short_leg.values()), 1.0)
    assert np.isclose(long_leg["a"] / long_leg["b"], 1.5)
    assert np.isclose(abs(short_leg["d"] / short_leg["c"]), 4.0)


def test_fundamental_ratio_information_is_residualized_not_raw() -> None:
    raw_ratio_like_features = {
        "gross_profit_ratio",
        "gross_profit_ratio_q",
        "gross_profit_ratio_ttm",
        "net_profit_ratio",
        "net_profit_ratio_q",
        "net_profit_ratio_ttm",
        "debt_assets_ratio",
        "debt_to_equity_ratio",
        "current_ratio",
        "cash_ratio",
        "equity_multiplier",
        "total_asset_turn_over",
        "inventory_turn_over",
        "account_turn_over",
        "current_asset_turn_over",
        "fix_asset_turn_over",
        "roe",
        "roe_q",
        "roe_ttm",
        "roa",
        "roa_q",
        "roa_ttm",
        "eps",
        "eps_q",
        "eps_ttm",
        "fcffps",
        "fcffps_q",
        "fcffps_ttm",
    }
    pair_names = {name for name, _, _ in RELATIVE_FUNDAMENTAL_PAIRS}

    assert raw_ratio_like_features.isdisjoint(MODEL_FINANCIAL_FEATURES)
    assert {
        "rel_gross_profit_q_to_operating_revenue_q",
        "rel_net_profit_q_to_operating_revenue_q",
        "rel_total_liability_to_total_assets",
        "rel_total_current_assets_to_total_current_liability",
        "rel_money_to_total_current_liability",
        "rel_operating_revenue_ttm_to_total_current_assets",
        "rel_operating_revenue_ttm_to_fixed_assets",
    }.issubset(pair_names)
    assert {"rel_total_assets_to_equity", "rel_total_assets_to_total_liability"}.isdisjoint(pair_names)


def test_fixed_model_feature_spec_is_redundancy_pruned_and_source_complete() -> None:
    assert len(MODEL_DAILY_FEATURES) == 20
    assert len(MODEL_FINANCIAL_FEATURES) == 38
    assert len(RELATIVE_FUNDAMENTAL_PAIRS) == 28
    assert MODEL_ANNOUNCEMENT_FEATURES == ["financial_announcement_recency_days"]
    assert "pv_turnover_6m" not in MODEL_DAILY_FEATURES
    assert {"total_profit", "total_profit_q", "total_profit_ttm"}.isdisjoint(MODEL_FINANCIAL_FEATURES)

    sources = set(FINANCIAL_SOURCE_CANDIDATES)
    assert set(MODEL_FINANCIAL_FEATURES).issubset(sources)
    daily_sources = {"nmv", "nshare"}
    for _, numerator, denominator in RELATIVE_FUNDAMENTAL_PAIRS:
        assert numerator in sources or numerator in daily_sources
        assert denominator in sources or denominator in daily_sources


def test_barra_style_exposures_use_residualized_fundamentals() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-31")] * 3,
            "z_nmv": [-1.0, 0.0, 1.0],
            "z_amount": [1.0, 2.0, 3.0],
            "z_volume": [1.0, 2.0, 3.0],
            "ret_lag_2": [0.01, 0.02, -0.01],
            "ret_lag_1": [0.02, -0.01, 0.03],
            "z_rel_net_profit_ttm_to_total_assets": [-1.0, 0.0, 1.0],
            "z_rel_gross_profit_ttm_to_operating_revenue_ttm": [-1.0, 0.0, 1.0],
            "z_rel_total_liability_to_total_assets": [1.0, 0.0, -1.0],
            "z_rel_total_liability_to_equity": [1.0, 0.0, -1.0],
        }
    )
    styled, style_cols = add_barra_style_exposures(frame, ["ret_lag_2", "ret_lag_1"])

    assert {"style_profitability", "style_leverage"}.issubset(style_cols)
    assert styled["style_profitability"].abs().sum() > 0
    assert styled["style_leverage"].abs().sum() > 0


def test_factor_distribution_tables_summarize_each_signal() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"]),
            "cmm": [-1.0, 1.0, -2.0, 2.0],
            "neutral": [-0.5, 0.5, -1.0, 1.0],
        }
    )
    values, summary, monthly, monthly_summary = factor_distribution_tables(
        frame,
        {"CMM": "cmm", "CMM Neutralized": "neutral"},
    )

    assert len(values) == 8
    assert set(summary["factor"]) == {"CMM", "CMM Neutralized"}
    assert set(monthly_summary["months"]) == {2}
    assert monthly.groupby("factor")["signal_date"].nunique().eq(2).all()


def test_style_exposure_regression_recovers_cross_sectional_betas() -> None:
    n = 60
    style_a = np.linspace(-1.0, 1.0, n)
    style_b = np.sin(np.linspace(0.0, 3.0, n))
    frame = pd.DataFrame(
        {
            "signal_date": pd.Timestamp("2024-01-31"),
            "factor": 2.0 * style_a - 1.5 * style_b,
            "style_a": style_a,
            "style_b": style_b,
        }
    )

    exposure = estimate_factor_style_exposure(frame, "factor", ["style_a", "style_b"])

    assert np.isclose(exposure.loc[0, "style_a"], 2.0)
    assert np.isclose(exposure.loc[0, "style_b"], -1.5)


def test_backtested_neutral_signal_remains_size_and_industry_neutral() -> None:
    rng = np.random.default_rng(7)
    n = 120
    size = np.linspace(-2.0, 2.0, n)
    industry = np.where(np.arange(n) % 2 == 0, "A", "B")
    cmm = 1.5 * size + (industry == "B") * 0.8 + rng.normal(0.0, 0.2, n)
    cmm[-1] = 100.0
    frame = pd.DataFrame(
        {
            "signal_date": pd.Timestamp("2024-01-31"),
            "cmm_signal_cs_z": cmm,
            "sm_signal": rng.normal(size=n),
            "z_nmv": size,
            "ind1": industry,
            "style_size": size,
        }
    )

    prepared, _, signals, _, _ = prepare_factor_signals(frame, ["style_size"])
    actual = signals["CMM Neutralized"]

    assert abs(prepared[actual].corr(prepared["z_nmv"])) < 1e-10
    assert prepared.groupby("ind1")[actual].mean().abs().max() < 1e-10


if __name__ == "__main__":
    test_signal_calendar_uses_next_trading_day_entry()
    test_monthly_winsorize_is_cross_sectional()
    test_worldquant_weights_are_demeaned_and_gross_normalized()
    test_fundamental_ratio_information_is_residualized_not_raw()
    test_fixed_model_feature_spec_is_redundancy_pruned_and_source_complete()
    test_barra_style_exposures_use_residualized_fundamentals()
    test_factor_distribution_tables_summarize_each_signal()
    print("All core logic tests passed.")
