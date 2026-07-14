from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "code"))

from model_compare_workflow import load_model_compare_data, prepare_factor_signals  # noqa: E402
from project_config import config_section  # noqa: E402


OUT_DIR = PROJECT_ROOT / "result" / "reports" / "style_exposure"


def estimate_factor_style_exposure(
    frame: pd.DataFrame,
    factor_col: str,
    style_cols: list[str],
    min_observations: int = 50,
) -> pd.DataFrame:
    rows = []
    for signal_date, month in frame.groupby("signal_date", sort=True):
        columns = [factor_col] + style_cols
        sample = month[columns].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sample) < max(min_observations, len(style_cols) + 2):
            continue
        x = np.column_stack([np.ones(len(sample)), sample[style_cols].to_numpy(dtype=float)])
        beta, *_ = np.linalg.lstsq(x, sample[factor_col].to_numpy(dtype=float), rcond=None)
        rows.append({"signal_date": pd.Timestamp(signal_date), **dict(zip(style_cols, beta[1:]))})
    return pd.DataFrame(rows, columns=["signal_date"] + style_cols)


def summarize_style_exposure(monthly: pd.DataFrame, style_cols: list[str]) -> pd.DataFrame:
    rows = []
    for style in style_cols:
        values = pd.to_numeric(monthly[style], errors="coerce").dropna()
        std = values.std(ddof=1)
        rows.append(
            {
                "style": style.removeprefix("style_"),
                "months": len(values),
                "mean_exposure": values.mean(),
                "mean_abs_exposure": values.abs().mean(),
                "exposure_std": std,
                "t_stat": values.mean() / (std / np.sqrt(len(values))) if len(values) > 1 and std > 0 else np.nan,
                "positive_rate": values.gt(0).mean() if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_split = str(config_section("backtest").get("eval_split", "test"))
    frame, _, _, style_cols = load_model_compare_data(PROJECT_ROOT, eval_split)
    frame, _, signals, _, _ = prepare_factor_signals(frame, style_cols)
    factor_col = signals["CMM Neutralized"]
    monthly = estimate_factor_style_exposure(frame, factor_col, style_cols)
    summary = summarize_style_exposure(monthly, style_cols)
    monthly.to_csv(OUT_DIR / "cmm_neutralized_style_exposure_monthly.csv", index=False)
    summary.to_csv(OUT_DIR / "cmm_neutralized_style_exposure_summary.csv", index=False)
    plot = summary.sort_values("mean_exposure")
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.barh(plot["style"], plot["mean_exposure"])
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_title("CMM Size-Industry Neutralized Factor: Mean Style Exposure")
    axis.set_xlabel("Mean monthly cross-sectional exposure")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cmm_neutralized_style_exposure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
