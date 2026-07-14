from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_group_and_long_short_nav(
    group_returns: pd.DataFrame,
    long_short_returns: pd.Series,
    output_path: Path,
    title: str,
) -> Path:
    group_nav = (1.0 + group_returns.fillna(0.0)).cumprod()
    long_short_nav = (1.0 + long_short_returns.fillna(0.0)).cumprod()
    fig, left = plt.subplots(figsize=(10, 5.5))
    for group in group_nav.columns:
        left.plot(group_nav.index, group_nav[group], linewidth=1.2, label=f"D{group}")
    left.set_xlabel("Date")
    left.set_ylabel("Equal-weight group cumulative NAV")
    right = left.twinx()
    right.plot(
        long_short_nav.index,
        long_short_nav.values,
        color="black",
        linestyle="--",
        linewidth=2.4,
        label="Full cross-section long-short",
    )
    right.set_ylabel("Long-short cumulative NAV")
    left_lines, left_labels = left.get_legend_handles_labels()
    right_lines, right_labels = right.get_legend_handles_labels()
    left.legend(left_lines + right_lines, left_labels + right_labels, ncol=4, fontsize=8, loc="upper left")
    left.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
