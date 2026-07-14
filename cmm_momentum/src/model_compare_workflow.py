from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "code"))

from backtest import (  # noqa: E402
    BacktestConfig,
    DailyMarketData,
    add_barra_style_exposures,
    load_trade_flags,
    neutralize_by_barra_style,
    neutralize_by_size_industry,
    plot_group_and_long_short_nav,
    prepare_trade_flags,
    run_monthly_factor_backtest,
)
from factor_processing import apply_monthly_winsorize  # noqa: E402
from project_config import config_section  # noqa: E402


OUT_DIR = PROJECT_ROOT / "result" / "reports" / "model_compare"
FACTOR_WINSOR_LIMITS = {
    "cmm_signal_cs_z": (0.01, 0.99),
}


def backtest_config() -> tuple[str, BacktestConfig]:
    values = config_section("backtest")
    eval_split = str(values.get("eval_split", "test"))
    if eval_split != "test" and not bool(values.get("allow_validation_split", False)):
        raise ValueError("Validation backtests require backtest.allow_validation_split=true")
    return eval_split, BacktestConfig(
        n_groups=int(values.get("deciles", 10)),
        gross=float(values.get("gross", 1.0)),
        max_abs_weight=float(values.get("max_abs_weight", 0.0)),
        transaction_cost_bps=float(values.get("transaction_cost_bps", 0.0)),
        min_ic_observations=int(values.get("min_ic_observations", 50)),
        periods_per_year=int(values.get("periods_per_year", 252)),
        risk_free_rate=float(values.get("risk_free_rate", 0.0)),
    )


def load_model_compare_data(
    project_root: Path,
    eval_split: str = "test",
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    data_path = project_root / "result" / "datasets" / "cmm_model_training_data.parquet"
    pred_path = project_root / "result" / "models" / "cmm" / "cmm_predictions.parquet"
    return_cols = (project_root / "result" / "datasets" / "cmm_return_window_columns.txt").read_text().splitlines()
    feature_cols = (project_root / "result" / "datasets" / "cmm_feature_columns.txt").read_text().splitlines()
    pred = pd.read_parquet(pred_path)
    pred["signal_date"] = pd.to_datetime(pred["signal_date"])
    base_cols = list(
        dict.fromkeys(
            ["stock_id", "signal_date", "trade_date", "exit_date", "target_1m_ret", "ind1"]
            + return_cols
            + feature_cols
        )
    )
    base = pd.read_parquet(data_path, columns=base_cols)
    for column in ["signal_date", "trade_date", "exit_date"]:
        base[column] = pd.to_datetime(base[column])
    base["ind1"] = base["ind1"].fillna("Unknown")
    base["sm_signal"] = base[return_cols].sum(axis=1)
    frame = pred[["stock_id", "signal_date", "split", "cmm_signal_cs_z", "target_1m_ret", "z_hat"]].merge(
        base[["stock_id", "signal_date", "trade_date", "exit_date", "sm_signal", "ind1"] + return_cols + feature_cols],
        on=["stock_id", "signal_date"],
        how="inner",
    )
    evaluation = frame.loc[frame["split"].eq(eval_split)].copy()
    if evaluation.duplicated(["stock_id", "signal_date"]).any():
        raise ValueError(f"Duplicate stock-month predictions detected in {eval_split} split")
    evaluation = prepare_trade_flags(evaluation, load_trade_flags(evaluation["trade_date"]))
    evaluation, style_cols = add_barra_style_exposures(evaluation, return_cols)
    return evaluation, return_cols, feature_cols, style_cols


def prepare_factor_signals(
    frame: pd.DataFrame,
    style_cols: list[str],
    factor_winsor_limits: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], pd.DataFrame, pd.DataFrame]:
    limits = factor_winsor_limits or FACTOR_WINSOR_LIMITS
    out = frame.copy()
    cmm_input = "cmm_signal_cs_z"
    if cmm_input in limits:
        cmm_input = "cmm_signal_cs_z_winsor"
        out[cmm_input] = apply_monthly_winsorize(out, "cmm_signal_cs_z", limits["cmm_signal_cs_z"])
    out["cmm_neutral"] = neutralize_by_size_industry(out, cmm_input)
    out["sm_neutral"] = neutralize_by_size_industry(out, "sm_signal")
    out["cmm_barra_neutral"] = neutralize_by_barra_style(out, cmm_input, style_cols=style_cols)
    raw_signals = {
        "CMM": "cmm_signal_cs_z",
        "Standard Momentum": "sm_signal",
        "CMM Neutralized": "cmm_neutral",
        "Standard Momentum Neutralized": "sm_neutral",
        "CMM Barra Neutralized": "cmm_barra_neutral",
    }
    signals = {
        "CMM": cmm_input,
        "Standard Momentum": "sm_signal",
        "CMM Neutralized": "cmm_neutral",
        "Standard Momentum Neutralized": "sm_neutral",
        "CMM Barra Neutralized": "cmm_barra_neutral",
    }
    size_industry_check = out[["cmm_signal_cs_z", "cmm_neutral", "sm_signal", "sm_neutral", "z_nmv"]].corr()
    barra_check = out[["cmm_signal_cs_z", "cmm_barra_neutral"] + style_cols].corr().loc[
        ["cmm_barra_neutral"], style_cols
    ]
    return out, raw_signals, signals, size_industry_check, barra_check


def run_factor_backtests(
    frame: pd.DataFrame,
    signals: dict[str, str],
    output_dir: Path,
    eval_split: str,
    config: BacktestConfig,
) -> dict[str, object]:
    market = DailyMarketData.from_directory(frame["trade_date"].min(), frame["exit_date"].max())
    results = {}
    performance_rows = []
    leg_rows = []
    annual_rows = []
    ic_rows = []
    monthly_ic_rows = []
    turnover_rows = []
    filenames = {
        "CMM": "cmm",
        "Standard Momentum": "standard_momentum",
        "CMM Neutralized": "cmm_neutralized",
        "Standard Momentum Neutralized": "standard_momentum_neutralized",
        "CMM Barra Neutralized": "cmm_barra_neutralized",
    }
    for factor, signal_col in signals.items():
        result = run_monthly_factor_backtest(frame, signal_col, market, config)
        results[factor] = result
        performance_rows.append({"factor": factor, **result.performance.loc["long_short"].to_dict()})
        for portfolio in ["long", "short", "long_short"]:
            leg_rows.append({"factor": factor, "portfolio": portfolio, **result.performance.loc[portfolio].to_dict()})
            annual = result.annual_performance[portfolio].copy()
            annual.insert(0, "portfolio", portfolio)
            annual.insert(0, "factor", factor)
            annual_rows.append(annual)
        ic_rows.append({"factor": factor, **result.ic_summary.to_dict()})
        monthly = result.monthly_ic.copy()
        monthly.insert(0, "factor", factor)
        monthly_ic_rows.append(monthly)
        diagnostic = result.turnover
        missing_returns = diagnostic["long_short_missing_returns"].sum()
        held_observations = diagnostic["long_short_held_stock_observations"].sum()
        turnover_rows.append(
            {
                "factor": factor,
                "mean_turnover": diagnostic["long_short_turnover"].mean(),
                "median_turnover": diagnostic["long_short_turnover"].median(),
                "mean_gross_exposure": diagnostic["long_short_gross_exposure"].mean(),
                "mean_net_exposure": diagnostic["long_short_net_exposure"].mean(),
                "mean_held_names": diagnostic["long_short_held_names"].mean(),
                "missing_return_observations": missing_returns,
                "held_stock_observations": held_observations,
                "missing_return_rate": missing_returns / held_observations if held_observations else float("nan"),
            }
        )
        plot_group_and_long_short_nav(
            result.group_returns,
            result.long_short_returns,
            output_dir / f"decile_nav_{filenames[factor]}_{eval_split}.png",
            f"{factor}: Equal-Weight Deciles and Full Cross-Section Long-Short ({eval_split})",
        )
    pd.DataFrame(performance_rows).to_csv(output_dir / f"performance_metrics_{eval_split}.csv", index=False)
    pd.DataFrame(leg_rows).to_csv(output_dir / f"portfolio_leg_performance_{eval_split}.csv", index=False)
    pd.concat(annual_rows, ignore_index=True).to_csv(output_dir / f"annual_performance_{eval_split}.csv", index=False)
    pd.DataFrame(ic_rows).to_csv(output_dir / f"factor_ic_summary_{eval_split}.csv", index=False)
    pd.concat(monthly_ic_rows, ignore_index=True).to_csv(output_dir / f"factor_monthly_ic_{eval_split}.csv", index=False)
    pd.DataFrame(turnover_rows).to_csv(output_dir / f"long_short_turnover_{eval_split}.csv", index=False)
    return results


def factor_distribution_tables(
    frame: pd.DataFrame,
    signal_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identifiers = [column for column in ["stock_id", "signal_date"] if column in frame.columns]
    parts = []
    for factor, column in signal_map.items():
        part = frame[identifiers + [column]].rename(columns={column: "signal"}).copy()
        part["factor"] = factor
        parts.append(part[identifiers + ["factor", "signal"]])
    values = pd.concat(parts, ignore_index=True)
    summary = values.groupby("factor", sort=False)["signal"].agg(
        count="count", mean="mean", std="std", min="min", max="max", skew="skew"
    ).reset_index()
    monthly = values.groupby(["factor", "signal_date"], sort=True)["signal"].agg(
        count="count", mean="mean", std="std", min="min", max="max", skew="skew"
    ).reset_index()
    monthly_summary = monthly.groupby("factor", sort=False).agg(
        months=("signal_date", "nunique"),
        avg_cs_mean=("mean", "mean"),
        avg_cs_std=("std", "mean"),
        avg_cs_skew=("skew", "mean"),
    ).reset_index()
    return values, summary, monthly, monthly_summary


def write_factor_distribution_outputs(frame: pd.DataFrame, signal_map: dict[str, str], output_dir: Path, eval_split: str) -> None:
    values, summary, monthly, monthly_summary = factor_distribution_tables(frame, signal_map)
    summary.to_csv(output_dir / f"cmm_factor_distribution_summary_{eval_split}.csv", index=False)
    monthly.to_csv(output_dir / f"cmm_factor_monthly_distribution_{eval_split}.csv", index=False)
    monthly_summary.to_csv(output_dir / f"cmm_factor_monthly_distribution_summary_{eval_split}.csv", index=False)
    fig, axes = plt.subplots(len(signal_map), 1, figsize=(9, 3.2 * len(signal_map)), squeeze=False)
    for axis, factor in zip(axes[:, 0], signal_map):
        signal = values.loc[values["factor"].eq(factor), "signal"].dropna()
        low, high = signal.quantile([0.001, 0.999])
        axis.hist(signal.clip(low, high), bins=60, alpha=0.85)
        axis.set_title(f"{factor} Cross-Sectional Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / f"cmm_factor_distribution_hist_{eval_split}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_split, config = backtest_config()
    frame, _, _, style_cols = load_model_compare_data(PROJECT_ROOT, eval_split)
    frame, raw_signals, signals, size_check, barra_check = prepare_factor_signals(frame, style_cols)
    results = run_factor_backtests(frame, signals, OUT_DIR, eval_split, config)
    write_factor_distribution_outputs(
        frame,
        {name: column for name, column in raw_signals.items() if name.startswith("CMM")},
        OUT_DIR,
        eval_split,
    )
    size_check.to_csv(OUT_DIR / f"neutralization_size_industry_check_{eval_split}.csv")
    barra_check.to_csv(OUT_DIR / f"neutralization_barra_check_{eval_split}.csv")
    print(pd.DataFrame({name: result.performance.loc["long_short"] for name, result in results.items()}).T)


if __name__ == "__main__":
    main()
