import numpy as np
import pandas as pd

from portfolio_optimization_system.backtest import rolling_oos_backtest


def sample_returns() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=550)
    rng = np.random.default_rng(99)
    data = rng.normal(0.0003, 0.012, size=(len(idx), 5))
    return pd.DataFrame(data, index=idx, columns=["US", "EU", "EM", "BOND", "GOLD"])


def test_backtest_runs() -> None:
    rets = sample_returns()
    assets = list(rets.columns)
    market_weights = pd.Series(np.ones(len(assets)) / len(assets), index=assets)
    p = pd.DataFrame([[1, -1, 0, 0, 0]], columns=assets, index=["view1"])
    q = pd.Series([0.02], index=p.index)
    conf = pd.Series([0.6], index=p.index)

    result = rolling_oos_backtest(
        returns=rets,
        train_window=252,
        rebalance_step=21,
        risk_aversion=3.0,
        long_only=True,
        market_weights=market_weights,
        bl_p=p,
        bl_q=q,
        bl_conf=conf,
        bl_tau=0.05,
    )

    assert not result.daily_returns.empty
    assert {"mean_variance", "risk_parity", "max_diversification", "black_litterman", "equal_weight"}.issubset(
        result.daily_returns.columns
    )
    assert not result.metrics.empty
