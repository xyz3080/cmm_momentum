from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from project_config import config_section


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


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
    relative_fundamental_min_obs: int = 50
    compression: str = "zstd"
    parquet_row_group_size: int = 50_000

    @classmethod
    def from_project_config(cls) -> "CleanConfig":
        data_cfg = config_section("data")
        allowed = {
            "start_date",
            "end_date",
            "lookback_start",
            "lookback_end",
            "min_history_non_null",
            "min_listing_days",
            "signal_entry_lag_days",
            "winsor_quantile",
            "relative_fundamental_min_obs",
            "compression",
            "parquet_row_group_size",
        }
        kwargs = {key: value for key, value in data_cfg.items() if key in allowed}
        return cls(**kwargs)

    @property
    def daily_dir(self) -> Path:
        return self.data_dir / "daily"

    @property
    def financial_path(self) -> Path:
        return self.data_dir / "financial" / "A_stock_financial.feather"
