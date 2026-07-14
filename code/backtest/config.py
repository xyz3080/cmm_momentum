from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    n_groups: int = 10
    gross: float = 1.0
    max_abs_weight: float = 0.0
    transaction_cost_bps: float = 0.0
    min_ic_observations: int = 20
    periods_per_year: int = 252
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        if self.gross <= 0:
            raise ValueError("gross must be positive")
        if self.max_abs_weight < 0:
            raise ValueError("max_abs_weight cannot be negative")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.min_ic_observations < 2:
            raise ValueError("min_ic_observations must be at least 2")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
