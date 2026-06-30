from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


DAILY_COLUMNS = [
    "date",
    "symbol",
    "name",
    "istd",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "ret",
    "preClose",
    "ind1",
    "tshare",
    "nshare",
    "tmv",
    "nmv",
    "st",
    "firstday",
    "ind2",
    "ind3",
    "sector",
    "ROE",
    "ROIC",
    "FCFF",
    "adjfactor",
]


FINANCIAL_FEATURE_CANDIDATES = [
    "total_assets",
    "total_liability",
    "total_shareholder_equity",
    "total_current_assets",
    "total_current_liability",
    "total_non_current_assets",
    "total_non_current_liability",
    "money",
    "inventories",
    "fixed_assets",
    "intangible_assets",
    "good_will",
    "account_receivable",
    "accounts_payable",
    "operating_revenue",
    "operating_revenue_q",
    "operating_revenue_ttm",
    "operating_cost",
    "operating_cost_q",
    "operating_cost_ttm",
    "operating_profit",
    "operating_profit_q",
    "operating_profit_ttm",
    "total_profit",
    "total_profit_q",
    "total_profit_ttm",
    "net_profit",
    "net_profit_q",
    "net_profit_ttm",
    "net_profit_cut",
    "net_profit_cut_q",
    "net_profit_cut_ttm",
    "gross_profit",
    "gross_profit_q",
    "gross_profit_ttm",
    "net_operate_cash_flow",
    "net_operate_cash_flow_q",
    "net_operate_cash_flow_ttm",
    "net_invest_cash_flow",
    "net_invest_cash_flow_q",
    "net_invest_cash_flow_ttm",
    "net_finance_cash_flow",
    "net_finance_cash_flow_q",
    "net_finance_cash_flow_ttm",
    "ebit",
    "ebit_q",
    "ebit_ttm",
    "ebitda",
    "ebitda_q",
    "ebitda_ttm",
    "eps",
    "eps_q",
    "eps_ttm",
    "roe",
    "roe_q",
    "roe_ttm",
    "roa",
    "roa_q",
    "roa_ttm",
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
    "oper_revenue_gr",
    "oper_revenue_gr_q",
    "oper_revenue_gr_ttm",
    "net_profit_gr",
    "net_profit_gr_q",
    "net_profit_gr_ttm",
    "eps_gr",
    "eps_gr_q",
    "eps_gr_ttm",
    "net_asset_gr",
    "current_assets_gr",
    "current_liability_gr",
    "inventory_gr",
    "fcff",
    "fcff_q",
    "fcff_ttm",
    "fcffps",
    "fcffps_q",
    "fcffps_ttm",
]


@dataclass(frozen=True)
class CleanConfig:
    data_dir: Path = WORKSPACE_ROOT / "data"
    result_dir: Path = PROJECT_ROOT / "result"
    start_date: str | None = None
    end_date: str | None = None
    lookback_start: int = 252
    lookback_end: int = 22
    min_history_non_null: int = 200
    min_listing_days: int = 365
    signal_entry_lag_days: int = 1
    winsor_quantile: float = 0.01
    compression: str = "zstd"

    @property
    def daily_dir(self) -> Path:
        return self.data_dir / "daily"

    @property
    def financial_path(self) -> Path:
        return self.data_dir / "financial" / "A_stock_financial.feather"


def normalize_daily_symbol(symbol: str) -> str:
    if pd.isna(symbol):
        return symbol
    symbol = str(symbol)
    if symbol.startswith("SH"):
        return f"{symbol[2:]}.SH"
    if symbol.startswith("SZ"):
        return f"{symbol[2:]}.SZ"
    if symbol.startswith("BJ"):
        return f"{symbol[2:]}.BJ"
    return symbol


def _date_from_daily_path(path: Path) -> pd.Timestamp:
    return pd.Timestamp(path.stem)


def list_daily_files(config: CleanConfig) -> list[Path]:
    files = sorted(config.daily_dir.glob("*.csv"))
    if config.start_date:
        start = pd.Timestamp(config.start_date)
        files = [path for path in files if _date_from_daily_path(path) >= start]
    if config.end_date:
        end = pd.Timestamp(config.end_date)
        files = [path for path in files if _date_from_daily_path(path) <= end]
    if not files:
        raise FileNotFoundError(f"No daily CSV files found under {config.daily_dir}")
    return files


def read_daily_panel(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        frame = pd.read_csv(path, usecols=lambda col: col in DAILY_COLUMNS)
        frames.append(frame)

    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["stock_id"] = daily["symbol"].map(normalize_daily_symbol)
    daily["firstday"] = pd.to_datetime(daily["firstday"], errors="coerce")
    daily["ret"] = pd.to_numeric(daily["ret"], errors="coerce")
    daily["log_ret"] = np.log1p(daily["ret"].clip(lower=-0.999999))

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "preClose",
        "tshare",
        "nshare",
        "tmv",
        "nmv",
        "ROE",
        "ROIC",
        "FCFF",
        "adjfactor",
    ]
    for col in numeric_cols:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")

    return daily.sort_values(["date", "stock_id"]).reset_index(drop=True)


def build_signal_calendar(trading_dates: pd.Series, entry_lag_days: int = 1) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(trading_dates).sort_values().unique(), name="date")
    calendar = pd.DataFrame({"date": dates})
    month_end_dates = calendar.groupby(calendar["date"].dt.to_period("M"))["date"].max()
    signal = pd.DataFrame({"signal_date": month_end_dates.values})
    signal["trade_date"] = signal["signal_date"].map(
        lambda d: _nth_next_trading_day(calendar["date"], d, entry_lag_days)
    )
    signal["next_signal_date"] = signal["signal_date"].shift(-1)
    signal["exit_date"] = signal["next_signal_date"].map(
        lambda d: _nth_next_trading_day(calendar["date"], d, entry_lag_days)
        if pd.notna(d)
        else pd.NaT
    )
    return signal.dropna().reset_index(drop=True)


def _nth_next_trading_day(dates: pd.Series, date: pd.Timestamp, n: int) -> pd.Timestamp | pd.NaT:
    pos = dates.searchsorted(pd.Timestamp(date), side="right")
    target = pos + n - 1
    if target >= len(dates):
        return pd.NaT
    return dates.iloc[target]


def make_return_windows(
    daily: pd.DataFrame,
    signal_calendar: pd.DataFrame,
    config: CleanConfig,
) -> pd.DataFrame:
    daily = daily.sort_values(["date", "stock_id"])
    ret_wide = daily.pivot_table(index="date", columns="stock_id", values="log_ret", aggfunc="last").sort_index()
    cum_ret = ret_wide.fillna(0.0).cumsum()
    signal_dates = set(signal_calendar["signal_date"])
    signal_rows = daily[daily["date"].isin(signal_dates)].copy()

    meta_cols = [
        "date",
        "stock_id",
        "symbol",
        "name",
        "istd",
        "close",
        "amount",
        "volume",
        "tmv",
        "nmv",
        "st",
        "firstday",
        "ind1",
        "ind2",
        "ind3",
        "sector",
        "ROE",
        "ROIC",
        "FCFF",
        "adjfactor",
    ]
    meta_cols = [col for col in meta_cols if col in signal_rows.columns]

    pieces = []
    lag_cols = [f"ret_lag_{lag}" for lag in range(config.lookback_start, config.lookback_end - 1, -1)]
    date_index = ret_wide.index

    for row in signal_calendar.itertuples(index=False):
        signal_date = pd.Timestamp(row.signal_date)
        trade_date = pd.Timestamp(row.trade_date)
        exit_date = pd.Timestamp(row.exit_date)

        pos = date_index.searchsorted(signal_date)
        start = pos - config.lookback_start
        stop = pos - config.lookback_end + 1
        if start < 0 or stop <= start:
            continue

        window = ret_wide.iloc[start:stop]
        if len(window) != len(lag_cols):
            continue

        window_by_stock = window.T
        window_by_stock.columns = lag_cols
        history_non_null = window_by_stock.notna().sum(axis=1)
        window_by_stock = window_by_stock.fillna(0.0)
        window_by_stock["history_non_null"] = history_non_null

        eligible = signal_rows.loc[signal_rows["date"] == signal_date, meta_cols].copy()
        eligible["listing_days"] = (signal_date - eligible["firstday"]).dt.days
        eligible = eligible[
            (eligible["istd"] == 1)
            & (eligible["st"] == 0)
            & (eligible["listing_days"] >= config.min_listing_days)
            & eligible["close"].notna()
        ]

        if trade_date not in cum_ret.index or exit_date not in cum_ret.index:
            continue

        target_log = (cum_ret.loc[exit_date] - cum_ret.loc[trade_date]).rename("target_1m_log_ret")
        target = np.expm1(target_log).rename("target_1m_ret")

        sample = eligible.merge(window_by_stock, left_on="stock_id", right_index=True, how="inner")
        sample = sample[sample["history_non_null"] >= config.min_history_non_null]
        sample = sample.merge(target_log, left_on="stock_id", right_index=True, how="left")
        sample = sample.merge(target, left_on="stock_id", right_index=True, how="left")
        sample = sample.dropna(subset=["target_1m_log_ret"])
        sample["signal_date"] = signal_date
        sample["trade_date"] = trade_date
        sample["exit_date"] = exit_date
        pieces.append(sample)

    if not pieces:
        raise RuntimeError("No monthly samples were generated. Check date range and filters.")

    samples = pd.concat(pieces, ignore_index=True)
    samples = samples.rename(columns={"date": "calendar_signal_date"})
    samples["target_1m_ret_cs_z"] = samples.groupby("signal_date")["target_1m_ret"].transform(_zscore)
    return samples


def select_financial_columns(financial_path: Path) -> list[str]:
    try:
        import pyarrow.feather as feather

        all_cols = feather.read_table(financial_path, memory_map=True).column_names
    except Exception:
        all_cols = pd.read_feather(financial_path, columns=None).columns
    required = ["stock_id", "stock_name", "report_date", "public_date"]
    features = [col for col in FINANCIAL_FEATURE_CANDIDATES if col in all_cols]
    return required + features


def read_financial_features(config: CleanConfig) -> tuple[pd.DataFrame, list[str]]:
    columns = select_financial_columns(config.financial_path)
    financial = pd.read_feather(config.financial_path, columns=columns)
    financial["report_date"] = pd.to_datetime(financial["report_date"])
    financial["public_date"] = pd.to_datetime(financial["public_date"])
    feature_cols = [col for col in columns if col not in {"stock_id", "stock_name", "report_date", "public_date"}]

    for col in feature_cols:
        financial[col] = pd.to_numeric(financial[col], errors="coerce")

    return financial.sort_values(["stock_id", "public_date", "report_date"]), feature_cols


def attach_point_in_time_financials(
    samples: pd.DataFrame,
    financial: pd.DataFrame,
    financial_feature_cols: list[str],
) -> pd.DataFrame:
    keep_cols = ["stock_id", "report_date", "public_date"] + financial_feature_cols
    right = financial[keep_cols].copy()
    right = right.sort_values(["stock_id", "public_date", "report_date"])
    right = right.drop_duplicates(["stock_id", "public_date"], keep="last")
    right = right.rename(
        columns={
            "report_date": "financial_report_date",
            "public_date": "financial_public_date",
        }
    )

    left = samples.sort_values(["signal_date", "stock_id"]).copy()
    right = right.sort_values(["financial_public_date", "stock_id"])
    merged = pd.merge_asof(
        left,
        right,
        left_on="signal_date",
        right_on="financial_public_date",
        by="stock_id",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


def preprocess_features(
    samples: pd.DataFrame,
    financial_feature_cols: list[str],
    config: CleanConfig,
) -> tuple[pd.DataFrame, list[str]]:
    # Keep only market/listing fields from daily files. Accounting fields in the
    # daily CSVs do not carry announcement dates, so the point-in-time financial
    # features below are the only accounting inputs used by the model.
    daily_feature_cols = [
        col
        for col in ["amount", "volume", "tmv", "nmv", "adjfactor", "listing_days"]
        if col in samples.columns
    ]
    feature_cols = daily_feature_cols + financial_feature_cols
    feature_cols = [col for col in feature_cols if col in samples.columns]

    cleaned = samples.copy()
    cleaned[feature_cols] = cleaned[feature_cols].replace([np.inf, -np.inf], np.nan)

    for col in feature_cols:
        cleaned[col] = cleaned.groupby("signal_date")[col].transform(
            lambda s: _winsorize_fill_zscore(s, config.winsor_quantile)
        )
        cleaned[col] = cleaned[col].fillna(0.0)

    rename_map = {col: f"z_{col}" for col in feature_cols}
    cleaned = cleaned.rename(columns=rename_map)
    return cleaned, [rename_map[col] for col in feature_cols]


def _winsorize_fill_zscore(series: pd.Series, q: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    lo = values.quantile(q)
    hi = values.quantile(1 - q)
    values = values.clip(lo, hi)
    values = values.fillna(values.median())
    return _zscore(values)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / std


def _project_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        pass
    return str(path)


def write_results(
    samples: pd.DataFrame,
    signal_calendar: pd.DataFrame,
    return_cols: list[str],
    feature_cols: list[str],
    config: CleanConfig,
) -> dict:
    dataset_dir = config.result_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / "cmm_model_training_data.parquet"
    calendar_path = dataset_dir / "cmm_signal_calendar.csv"
    return_cols_path = dataset_dir / "cmm_return_window_columns.txt"
    feature_cols_path = dataset_dir / "cmm_feature_columns.txt"
    summary_path = dataset_dir / "data_clean_summary.json"

    samples.to_parquet(dataset_path, index=False, compression=config.compression)
    signal_calendar.to_csv(calendar_path, index=False)
    return_cols_path.write_text("\n".join(return_cols) + "\n", encoding="utf-8")
    feature_cols_path.write_text("\n".join(feature_cols) + "\n", encoding="utf-8")

    summary = {
        "dataset_path": _project_relative_path(dataset_path),
        "rows": int(len(samples)),
        "columns": int(samples.shape[1]),
        "stocks": int(samples["stock_id"].nunique()),
        "signal_months": int(samples["signal_date"].nunique()),
        "signal_date_min": str(pd.to_datetime(samples["signal_date"]).min().date()),
        "signal_date_max": str(pd.to_datetime(samples["signal_date"]).max().date()),
        "return_window_columns": len(return_cols),
        "feature_columns": len(feature_cols),
        "target_columns": ["target_1m_log_ret", "target_1m_ret", "target_1m_ret_cs_z"],
        "notes": [
            "Each row is a stock-month sample.",
            "Return window uses daily log returns from t-252 through t-22, matching the paper's 12-to-1 month formation window.",
            "Financial features are point-in-time: public_date <= signal_date.",
            "Features are winsorized, median-filled, and cross-sectionally z-scored within each signal_date.",
            "target_1m_ret_cs_z is the cross-sectional standardized training label.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_pipeline(config: CleanConfig | None = None) -> tuple[pd.DataFrame, dict]:
    config = config or CleanConfig()
    files = list_daily_files(config)
    print(f"Reading {len(files)} daily files...")
    daily = read_daily_panel(files)
    print(f"Daily panel: {daily.shape[0]:,} rows, {daily.shape[1]:,} columns")

    signal_calendar = build_signal_calendar(daily["date"], config.signal_entry_lag_days)
    signal_calendar = signal_calendar[
        (signal_calendar["signal_date"] >= daily["date"].min())
        & (signal_calendar["exit_date"] <= daily["date"].max())
    ].reset_index(drop=True)
    print(f"Signal months: {len(signal_calendar):,}")

    print("Building return windows and forward labels...")
    samples = make_return_windows(daily, signal_calendar, config)
    return_cols = [f"ret_lag_{lag}" for lag in range(config.lookback_start, config.lookback_end - 1, -1)]
    print(f"Samples before financial merge: {samples.shape[0]:,} rows")

    print("Reading financial features...")
    financial, financial_feature_cols = read_financial_features(config)
    print(f"Financial feature columns selected: {len(financial_feature_cols):,}")

    print("Attaching point-in-time financial features...")
    samples = attach_point_in_time_financials(samples, financial, financial_feature_cols)

    print("Preprocessing features...")
    samples, feature_cols = preprocess_features(samples, financial_feature_cols, config)

    id_cols = [
        "stock_id",
        "symbol",
        "name",
        "signal_date",
        "trade_date",
        "exit_date",
        "financial_report_date",
        "financial_public_date",
        "ind1",
        "ind2",
        "ind3",
        "sector",
        "close",
        "target_1m_log_ret",
        "target_1m_ret",
        "target_1m_ret_cs_z",
        "history_non_null",
    ]
    id_cols = [col for col in id_cols if col in samples.columns]
    final_cols = id_cols + return_cols + feature_cols
    samples = samples[final_cols].sort_values(["signal_date", "stock_id"]).reset_index(drop=True)

    print("Writing results...")
    summary = write_results(samples, signal_calendar, return_cols, feature_cols, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return samples, summary


if __name__ == "__main__":
    run_pipeline()
