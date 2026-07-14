"""Compatibility imports for older research scripts.

New code should add the repository's ``code`` directory to ``sys.path`` and import from ``backtest``.
"""

from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from backtest import (  # noqa: E402,F401
    BARRA_STYLE_COLS,
    BacktestConfig,
    BacktestResult,
    DailyMarketData,
    add_barra_style_exposures,
    annual_performance,
    build_signed_weights,
    build_worldquant_weights,
    cs_zscore,
    ic_summary,
    limit_rate,
    load_signal_market_caps,
    load_trade_flags,
    monthly_ic,
    neutralize_by_barra_style,
    neutralize_by_size_industry,
    normalize_daily_symbol,
    performance_stats,
    plot_group_and_long_short_nav,
    prepare_trade_flags,
    rebalance_signed_portfolio,
    run_monthly_factor_backtest,
    split_signed_weights,
)
