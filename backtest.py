from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .optimizers import (
    black_litterman_weights,
    max_diversification_weights,
    mean_variance_weights,
    risk_parity_weights,
)


@dataclass
class BacktestResult:
    daily_returns: pd.DataFrame
    weights_history: dict[str, pd.DataFrame]
    metrics: pd.DataFrame


def _compute_metrics(returns: pd.Series, trading_days: int = 252) -> dict[str, float]:
    if returns.empty:
        return {
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "total_return": np.nan,
        }

    cumulative = (1.0 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)

    n = len(returns)
    annual_return = float((1.0 + total_return) ** (trading_days / n) - 1.0)
    annual_vol = float(returns.std(ddof=1) * np.sqrt(trading_days))
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "total_return": total_return,
    }


def _slice_rebalance_indices(index: pd.DatetimeIndex, train_window: int, rebalance_step: int) -> list[int]:
    points = []
    i = train_window
    while i < len(index):
        points.append(i)
        i += rebalance_step
    return points


def rolling_oos_backtest(
    returns: pd.DataFrame,
    train_window: int,
    rebalance_step: int,
    risk_aversion: float,
    long_only: bool,
    market_weights: pd.Series,
    bl_p: pd.DataFrame,
    bl_q: pd.Series,
    bl_conf: pd.Series,
    bl_tau: float,
) -> BacktestResult:
    if train_window <= 20:
        raise ValueError("train_window must be > 20")

    idx = returns.index
    rebalance_points = _slice_rebalance_indices(idx, train_window, rebalance_step)

    strategy_returns: dict[str, pd.Series] = {
        "mean_variance": pd.Series(index=idx, dtype=float),
        "risk_parity": pd.Series(index=idx, dtype=float),
        "max_diversification": pd.Series(index=idx, dtype=float),
        "black_litterman": pd.Series(index=idx, dtype=float),
        "equal_weight": pd.Series(index=idx, dtype=float),
    }
    weights_history: dict[str, list[pd.Series]] = {k: [] for k in strategy_returns}

    for start_i in rebalance_points:
        train = returns.iloc[start_i - train_window : start_i]
        end_i = min(start_i + rebalance_step, len(idx))
        test = returns.iloc[start_i:end_i]
        if test.empty:
            continue

        w_mv = mean_variance_weights(train, risk_aversion=risk_aversion, long_only=long_only)
        w_rp = risk_parity_weights(train, long_only=long_only)
        w_md = max_diversification_weights(train, long_only=long_only)
        w_bl = black_litterman_weights(
            returns=train,
            market_weights=market_weights,
            p=bl_p,
            q=bl_q,
            confidence=bl_conf,
            tau=bl_tau,
            risk_aversion=risk_aversion,
            long_only=long_only,
        )
        w_eq = pd.Series(np.ones(train.shape[1]) / train.shape[1], index=train.columns, name="equal_weight")

        weight_map = {
            "mean_variance": w_mv,
            "risk_parity": w_rp,
            "max_diversification": w_md,
            "black_litterman": w_bl,
            "equal_weight": w_eq,
        }

        for strat, w in weight_map.items():
            daily = test.mul(w, axis=1).sum(axis=1)
            strategy_returns[strat].loc[daily.index] = daily.values

            snapshot = w.copy()
            snapshot.name = idx[start_i]
            weights_history[strat].append(snapshot)

    daily_returns_df = pd.DataFrame({k: v.dropna() for k, v in strategy_returns.items()})

    metric_rows = []
    for strat in daily_returns_df.columns:
        m = _compute_metrics(daily_returns_df[strat])
        m["strategy"] = strat
        metric_rows.append(m)
    metrics = pd.DataFrame(metric_rows).set_index("strategy").sort_values("sharpe", ascending=False)

    out_weights = {
        strat: (pd.DataFrame(items) if items else pd.DataFrame(columns=returns.columns))
        for strat, items in weights_history.items()
    }

    return BacktestResult(
        daily_returns=daily_returns_df,
        weights_history=out_weights,
        metrics=metrics,
    )
