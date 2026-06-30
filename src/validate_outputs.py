from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = PROJECT_ROOT / "result"


REQUIRED_FILES = [
    RESULT / "datasets" / "cmm_model_training_data.parquet",
    RESULT / "datasets" / "cmm_feature_columns.txt",
    RESULT / "datasets" / "cmm_return_window_columns.txt",
    RESULT / "datasets" / "cmm_signal_calendar.csv",
    RESULT / "models" / "cmm" / "cmm_predictions.parquet",
    RESULT / "models" / "cmm" / "cmm_model.pt",
    RESULT / "models" / "cmm" / "cmm_folds.csv",
    RESULT / "reports" / "model_compare" / "performance_metrics_test.csv",
    RESULT / "reports" / "model_compare" / "performance_metrics_value_weighted_test.csv",
    RESULT / "reports" / "cmm_explain" / "explanation_summary.csv",
    RESULT / "reports" / "barra_attribution" / "cmm_barra_attribution_summary.csv",
]


def check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)
    print(f"OK {message}")


def main() -> None:
    for path in REQUIRED_FILES:
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
    check({"CMM", "Standard Momentum", "CMM Neutralized", "Standard Momentum Neutralized"}.issubset(set(perf["factor"])), "main performance table has expected factors")

    print("All result validation checks passed.")


if __name__ == "__main__":
    main()

