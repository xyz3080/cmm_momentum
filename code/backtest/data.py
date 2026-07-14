from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DAILY_DIR = WORKSPACE_ROOT / "data" / "daily"


def normalize_daily_symbol(symbol: pd.Series) -> pd.Series:
    values = symbol.astype(str).str.strip()
    exchange = values.str[:2]
    code = values.str[2:]
    suffix = exchange.map({"SH": ".SH", "SZ": ".SZ", "BJ": ".BJ"}).fillna("")
    return code + suffix


def limit_rate(stock_id: pd.Series, trade_date: pd.Series, st: pd.Series | None = None) -> pd.Series:
    stock = stock_id.astype(str)
    date = pd.to_datetime(trade_date)
    rate = pd.Series(0.10, index=stock.index)
    rate[stock.str.startswith(("688", "300")) & date.ge(pd.Timestamp("2020-08-24"))] = 0.20
    rate[stock.str.startswith(("8", "4"))] = 0.30
    if st is not None:
        rate[pd.to_numeric(st, errors="coerce").fillna(0).astype(int).eq(1)] = 0.05
    return rate


def load_trade_flags(trade_dates: pd.Series, daily_dir: Path = DAILY_DIR) -> pd.DataFrame:
    pieces = []
    for date in sorted(pd.to_datetime(trade_dates.dropna().unique())):
        path = daily_dir / f"{date:%Y-%m-%d}.csv"
        if not path.exists():
            continue
        daily = pd.read_csv(path, usecols=["date", "symbol", "close", "preClose", "ret", "st"])
        daily["trade_date"] = pd.to_datetime(daily["date"])
        daily["stock_id"] = normalize_daily_symbol(daily["symbol"])
        rate = limit_rate(daily["stock_id"], daily["trade_date"], daily["st"])
        ret = pd.to_numeric(daily["ret"], errors="coerce")
        close = pd.to_numeric(daily["close"], errors="coerce")
        pre_close = pd.to_numeric(daily["preClose"], errors="coerce")
        daily["limit_up"] = ret.ge(rate - 0.002) | close.ge(pre_close * (1 + rate) * 0.998)
        daily["limit_down"] = ret.le(-rate + 0.002) | close.le(pre_close * (1 - rate) * 1.002)
        daily["is_tradable"] = True
        pieces.append(daily[["stock_id", "trade_date", "limit_up", "limit_down", "is_tradable"]])
    if not pieces:
        return pd.DataFrame(columns=["stock_id", "trade_date", "limit_up", "limit_down", "is_tradable"])
    return pd.concat(pieces, ignore_index=True).drop_duplicates(["stock_id", "trade_date"])


def prepare_trade_flags(frame: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    flag_cols = ["stock_id", "trade_date", "limit_up", "limit_down", "is_tradable"]
    base = frame.drop(columns=[col for col in flag_cols[2:] if col in frame.columns]).copy()
    merged = base.merge(flags[flag_cols], on=["stock_id", "trade_date"], how="left")
    merged["is_tradable"] = merged["is_tradable"].eq(True)
    merged["limit_up"] = merged["limit_up"].eq(True)
    merged["limit_down"] = merged["limit_down"].eq(True)
    merged["can_buy"] = merged["is_tradable"] & ~merged["limit_up"]
    merged["can_sell"] = merged["is_tradable"] & ~merged["limit_down"]
    return merged


def load_signal_market_caps(signal_dates: pd.Series, daily_dir: Path = DAILY_DIR) -> pd.DataFrame:
    pieces = []
    for date in sorted(pd.to_datetime(signal_dates.dropna().unique())):
        path = daily_dir / f"{date:%Y-%m-%d}.csv"
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


@dataclass(frozen=True)
class DailyMarketData:
    dates: tuple[pd.Timestamp, ...]
    returns_by_date: dict[pd.Timestamp, dict[str, float]]
    available_by_date: dict[pd.Timestamp, frozenset[str]]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "DailyMarketData":
        required = {"date", "stock_id", "ret"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"daily market data missing columns: {sorted(missing)}")
        clean = frame[list(required)].copy()
        clean["date"] = pd.to_datetime(clean["date"])
        clean["ret"] = pd.to_numeric(clean["ret"], errors="coerce")
        clean = clean.drop_duplicates(["date", "stock_id"], keep="last")
        returns = {}
        available = {}
        for date, group in clean.groupby("date", sort=True):
            key = pd.Timestamp(date)
            available[key] = frozenset(group["stock_id"].astype(str))
            values = group.dropna(subset=["ret"]).set_index("stock_id")["ret"].astype(float)
            returns[key] = values.replace([np.inf, -np.inf], np.nan).dropna().to_dict()
        dates = tuple(sorted(returns))
        return cls(dates=dates, returns_by_date=returns, available_by_date=available)

    @classmethod
    def from_directory(
        cls,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        daily_dir: Path = DAILY_DIR,
    ) -> "DailyMarketData":
        pieces = []
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        for path in sorted(daily_dir.glob("*.csv")):
            try:
                date = pd.Timestamp(path.stem)
            except ValueError:
                continue
            if not start <= date <= end:
                continue
            daily = pd.read_csv(path, usecols=["date", "symbol", "ret"])
            daily["date"] = pd.to_datetime(daily["date"])
            daily["stock_id"] = normalize_daily_symbol(daily["symbol"])
            pieces.append(daily[["date", "stock_id", "ret"]])
        if not pieces:
            return cls((), {}, {})
        return cls.from_frame(pd.concat(pieces, ignore_index=True))

    def period_dates(self, trade_date: pd.Timestamp, exit_date: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
        start = pd.Timestamp(trade_date)
        end = pd.Timestamp(exit_date)
        return tuple(date for date in self.dates if start < date <= end)
