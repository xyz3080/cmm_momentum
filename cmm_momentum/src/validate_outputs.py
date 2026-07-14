from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch

from research_metadata import feature_list_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = PROJECT_ROOT / "result"


CORE_REQUIRED_FILES = [
    RESULT / "datasets" / "cmm_model_training_data.parquet",
    RESULT / "datasets" / "cmm_feature_columns.txt",
    RESULT / "datasets" / "cmm_return_window_columns.txt",
    RESULT / "datasets" / "cmm_signal_calendar.csv",
    RESULT / "models" / "cmm" / "cmm_predictions.parquet",
    RESULT / "models" / "cmm" / "cmm_model.pt",
    RESULT / "models" / "cmm" / "cmm_folds.csv",
    RESULT / "reports" / "model_compare" / "performance_metrics_test.csv",
    RESULT / "reports" / "model_compare" / "factor_ic_summary_test.csv",
    RESULT / "reports" / "model_compare" / "portfolio_leg_performance_test.csv",
    RESULT / "reports" / "cmm_explain" / "explanation_summary.csv",
    RESULT / "reports" / "style_exposure" / "cmm_neutralized_style_exposure_summary.csv",
]

def check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)
    print(f"OK {message}")


def main() -> None:
    for path in CORE_REQUIRED_FILES:
        check(path.exists(), f"required file exists: {path.relative_to(PROJECT_ROOT)}")

    dataset_cols = [
        "stock_id",
        "signal_date",
        "trade_date",
        "exit_date",
        "financial_report_date",
        "financial_public_date",
        "target_1m_ret",
        "target_1m_ret_cs_z",
    ]
    train = pd.read_parquet(RESULT / "datasets" / "cmm_model_training_data.parquet", columns=dataset_cols)
    pred = pd.read_parquet(RESULT / "models" / "cmm" / "cmm_predictions.parquet")
    train_columns = pq.read_schema(RESULT / "datasets" / "cmm_model_training_data.parquet").names
    feature_cols = (RESULT / "datasets" / "cmm_feature_columns.txt").read_text(encoding="utf-8").splitlines()
    feature_hash = feature_list_hash(feature_cols)
    model = torch.load(RESULT / "models" / "cmm" / "cmm_model.pt", map_location="cpu")

    check("z_good_will" not in feature_cols, "feature list excludes z_good_will")
    check(
        {"z_tmv", "z_adjfactor", "z_listing_days"}.isdisjoint(feature_cols),
        "feature list excludes removed market features",
    )
    check(
        {"z_pv_mom_12m", "z_pv_vol_12m", "z_pv_turnover_12m"}.issubset(set(feature_cols)),
        "feature list includes historical price-volume features",
    )
    check(
        {"z_rel_net_profit_ttm_to_nmv", "z_rel_net_operate_cash_flow_q_to_net_profit_q"}.issubset(set(feature_cols)),
        "feature list includes relative fundamental residual features",
    )
    check(
        "z_financial_announcement_recency_days" in feature_cols,
        "feature list includes point-in-time financial announcement recency",
    )
    raw_fundamental_ratio_features = {
        "z_gross_profit_ratio",
        "z_gross_profit_ratio_q",
        "z_gross_profit_ratio_ttm",
        "z_net_profit_ratio",
        "z_net_profit_ratio_q",
        "z_net_profit_ratio_ttm",
        "z_debt_assets_ratio",
        "z_debt_to_equity_ratio",
        "z_current_ratio",
        "z_cash_ratio",
        "z_equity_multiplier",
        "z_total_asset_turn_over",
        "z_inventory_turn_over",
        "z_account_turn_over",
        "z_current_asset_turn_over",
        "z_fix_asset_turn_over",
        "z_roe",
        "z_roe_q",
        "z_roe_ttm",
        "z_roa",
        "z_roa_q",
        "z_roa_ttm",
        "z_eps",
        "z_eps_q",
        "z_eps_ttm",
        "z_fcffps",
        "z_fcffps_q",
        "z_fcffps_ttm",
    }
    check(
        raw_fundamental_ratio_features.isdisjoint(feature_cols),
        "feature list excludes raw fundamental ratio-like features",
    )
    check(
        {
            "z_rel_gross_profit_q_to_operating_revenue_q",
            "z_rel_net_profit_q_to_operating_revenue_q",
            "z_rel_total_liability_to_total_assets",
            "z_rel_total_current_assets_to_total_current_liability",
            "z_rel_money_to_total_current_liability",
            "z_rel_operating_revenue_ttm_to_fixed_assets",
        }.issubset(set(feature_cols)),
        "feature list includes residualized replacements for fundamental ratios",
    )
    price_volume = [
        col for col in feature_cols if col.startswith("z_pv_") or col in {"z_amount", "z_volume", "z_nmv"}
    ]
    relative_fundamental = [col for col in feature_cols if col.startswith("z_rel_")]
    announcement_timing = [col for col in feature_cols if col == "z_financial_announcement_recency_days"]
    raw_financial = [
        col for col in feature_cols if col not in set(price_volume + relative_fundamental + announcement_timing)
    ]
    check(
        (
            len(feature_cols),
            len(price_volume),
            len(raw_financial),
            len(relative_fundamental),
            len(announcement_timing),
        )
        == (87, 20, 38, 28, 1),
        "training feature list has expected 87 features split 20/38/28/1",
    )
    check(
        {
            "z_pv_turnover_6m",
            "z_total_profit",
            "z_total_profit_q",
            "z_total_profit_ttm",
            "z_rel_total_assets_to_total_liability",
        }.isdisjoint(feature_cols),
        "feature list excludes fixed high-correlation representatives",
    )
    check("z_good_will" not in train_columns, "training dataset excludes z_good_will")
    check(model["feature_cols"] == feature_cols, "model feature list matches dataset feature list")
    if "feature_hash" in model:
        check(model["feature_hash"] == feature_hash, "model feature hash matches dataset feature list")
    else:
        print("WARN model has no feature_hash metadata; retrain to enable strict feature-version validation")

    for col in ["signal_date", "trade_date", "exit_date", "financial_report_date", "financial_public_date"]:
        train[col] = pd.to_datetime(train[col])
    pred["signal_date"] = pd.to_datetime(pred["signal_date"])

    check((train["trade_date"] > train["signal_date"]).all(), "trade_date is after signal_date")
    check((train["exit_date"] > train["trade_date"]).all(), "exit_date is after trade_date")
    check((train["financial_public_date"] <= train["signal_date"]).all(), "financial public_date is point-in-time")
    check((train["financial_report_date"] <= train["financial_public_date"]).all(), "financial report_date is not after public_date")
    check(not train.duplicated(["stock_id", "signal_date"]).any(), "training data has no duplicate stock-month rows")

    test = pred[pred["split"].eq("test")].copy()
    check(len(test) > 0, "test predictions are present")
    check(not test.duplicated(["stock_id", "signal_date"]).any(), "test predictions have no duplicate stock-month rows")

    folds = pd.read_csv(RESULT / "models" / "cmm" / "cmm_folds.csv")
    check(folds["fold"].is_monotonic_increasing, "fold ids are ordered")
    check(set(pred["split"].unique()).issuperset({"test"}), "prediction split includes test")

    perf = pd.read_csv(RESULT / "reports" / "model_compare" / "performance_metrics_test.csv")
    check(
        {"CMM", "Standard Momentum", "CMM Neutralized", "Standard Momentum Neutralized", "CMM Barra Neutralized"}.issubset(
            set(perf["factor"])
        ),
        "main performance table has expected factors",
    )
    check(
        {"annual_return", "annualized_mean_return", "annual_vol", "sharpe", "max_drawdown"}.issubset(perf.columns),
        "performance table separates CAGR and annualized arithmetic mean",
    )
    turnover = pd.read_csv(RESULT / "reports" / "model_compare" / "long_short_turnover_test.csv")
    check(
        {"missing_return_observations", "held_stock_observations", "missing_return_rate"}.issubset(turnover.columns),
        "turnover diagnostics include the missing-return numerator, denominator and rate",
    )
    check(turnover["missing_return_rate"].between(0, 1).all(), "missing-return rates are valid probabilities")
    factor_style_exposure = pd.read_csv(
        RESULT / "reports" / "style_exposure" / "cmm_neutralized_style_exposure_summary.csv"
    )
    check(
        {"size", "liquidity", "momentum", "volatility", "profitability", "growth", "leverage"}.issubset(
            set(factor_style_exposure["style"])
        ),
        "CMM neutralized style exposure includes expected style factors",
    )
    for style in ["profitability", "leverage"]:
        exposure = factor_style_exposure.loc[
            factor_style_exposure["style"].eq(style), "mean_abs_exposure"
        ]
        check(
            len(exposure) == 1 and exposure.iloc[0] > 1e-6,
            f"{style} exposure uses active residualized fundamental inputs",
        )

    print("All result validation checks passed.")


if __name__ == "__main__":
    main()
