from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import BacktestResult


def _fmt(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.2%}"


def save_backtest_outputs(
    result: BacktestResult,
    output_dir: Path,
    symbol_group: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{symbol_group}_oos_metrics.csv"
    returns_path = output_dir / f"{symbol_group}_oos_daily_returns.csv"
    result.metrics.to_csv(metrics_path)
    result.daily_returns.to_csv(returns_path)

    weight_paths: dict[str, Path] = {}
    for strat, wdf in result.weights_history.items():
        p = output_dir / f"{symbol_group}_weights_{strat}.csv"
        wdf.to_csv(p)
        weight_paths[strat] = p

    return {
        "metrics": metrics_path,
        "returns": returns_path,
        **{f"weights_{k}": v for k, v in weight_paths.items()},
    }


def plot_cumulative_returns(result: BacktestResult, output_dir: Path, symbol_group: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    for col in result.daily_returns.columns:
        curve = (1.0 + result.daily_returns[col]).cumprod()
        ax.plot(curve.index, curve.values, label=col)

    ax.set_title(f"{symbol_group} Out-of-Sample Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = output_dir / f"{symbol_group}_oos_cumulative_returns.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def generate_markdown_report(
    result: BacktestResult,
    output_dir: Path,
    symbol_group: str,
    cumulative_plot_path: Path,
) -> Path:
    metrics = result.metrics.copy()
    best = metrics.index[0] if not metrics.empty else "NA"

    lines: list[str] = [
        f"# Portfolio Optimization OOS Report ({symbol_group})",
        "",
        "## Ranking (by Sharpe)",
    ]

    for strat, row in metrics.iterrows():
        lines.append(
            f"- {strat}: sharpe={row['sharpe']:.3f}, ann_return={_fmt(row['annual_return'])}, ann_vol={_fmt(row['annual_vol'])}, max_dd={_fmt(row['max_drawdown'])}"
        )

    lines.extend(
        [
            "",
            "## Key Takeaway",
            f"- Best out-of-sample Sharpe strategy: `{best}`",
            "- Compare this with max drawdown and total return before selecting a production allocation rule.",
            "",
            "## Plot",
            f"- Cumulative returns: `{cumulative_plot_path.name}`",
        ]
    )

    report_path = output_dir / f"{symbol_group}_oos_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
