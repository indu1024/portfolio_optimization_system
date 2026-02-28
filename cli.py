from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import rolling_oos_backtest
from .data_loader import load_bl_views, load_market_weights, load_prices_from_csv, returns_from_prices
from .report import generate_markdown_report, plot_cumulative_returns, save_backtest_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Portfolio optimization system with mean-variance, risk parity, max diversification, "
            "and Black-Litterman plus out-of-sample comparison."
        )
    )
    parser.add_argument("--prices-csv", type=Path, required=True, help="CSV with Date + asset price columns")
    parser.add_argument(
        "--returns-method",
        choices=["log", "simple"],
        default="simple",
        help="Return transformation for backtest",
    )
    parser.add_argument(
        "--market-weights-csv",
        type=Path,
        help="Optional BL market weights CSV with columns asset,weight",
    )
    parser.add_argument(
        "--views-csv",
        type=Path,
        help="Optional BL views CSV (view,asset,loading,view_return,confidence)",
    )
    parser.add_argument("--train-window", type=int, default=252, help="In-sample training days per rebalance")
    parser.add_argument("--rebalance-step", type=int, default=21, help="Rebalance frequency in trading days")
    parser.add_argument("--risk-aversion", type=float, default=3.0, help="Risk aversion for utility-based methods")
    parser.add_argument("--bl-tau", type=float, default=0.05, help="Black-Litterman tau parameter")
    parser.add_argument("--allow-short", action="store_true", help="Allow short weights")
    parser.add_argument("--symbol-group", default="multi_asset", help="Output filename prefix")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    return parser.parse_args()


def _default_views(assets: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if len(assets) < 2:
        raise ValueError("Need at least 2 assets for default BL views")

    # Conservative default: first asset expected to outperform second by 2% annualized.
    p = pd.DataFrame([np.zeros(len(assets))], columns=assets, index=["default_view"])
    p.loc["default_view", assets[0]] = 1.0
    p.loc["default_view", assets[1]] = -1.0

    q = pd.Series([0.02], index=p.index)
    conf = pd.Series([0.6], index=p.index)
    return p, q, conf


def main() -> None:
    args = parse_args()

    prices = load_prices_from_csv(args.prices_csv)
    returns = returns_from_prices(prices, method=args.returns_method)

    assets = list(returns.columns)

    if args.market_weights_csv:
        market_weights = load_market_weights(args.market_weights_csv, assets)
    else:
        market_weights = pd.Series(np.ones(len(assets)) / len(assets), index=assets)

    if args.views_csv:
        bl_p, bl_q, bl_conf = load_bl_views(args.views_csv, assets)
    else:
        bl_p, bl_q, bl_conf = _default_views(assets)

    result = rolling_oos_backtest(
        returns=returns,
        train_window=args.train_window,
        rebalance_step=args.rebalance_step,
        risk_aversion=args.risk_aversion,
        long_only=not args.allow_short,
        market_weights=market_weights,
        bl_p=bl_p,
        bl_q=bl_q,
        bl_conf=bl_conf,
        bl_tau=args.bl_tau,
    )

    output_paths = save_backtest_outputs(
        result=result,
        output_dir=args.output_dir,
        symbol_group=args.symbol_group,
    )
    plot_path = plot_cumulative_returns(result=result, output_dir=args.output_dir, symbol_group=args.symbol_group)
    report_path = generate_markdown_report(
        result=result,
        output_dir=args.output_dir,
        symbol_group=args.symbol_group,
        cumulative_plot_path=plot_path,
    )

    print("Saved files:")
    print(f"- metrics: {output_paths['metrics']}")
    print(f"- oos daily returns: {output_paths['returns']}")
    print(f"- cumulative plot: {plot_path}")
    print(f"- markdown report: {report_path}")
    print("\nTop strategies by Sharpe:")
    print(result.metrics.to_string())


if __name__ == "__main__":
    main()
