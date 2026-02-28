import numpy as np
import pandas as pd

from portfolio_optimization_system.optimizers import (
    black_litterman_weights,
    max_diversification_weights,
    mean_variance_weights,
    risk_parity_weights,
)


def sample_returns() -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=300)
    rng = np.random.default_rng(123)
    data = rng.normal(0.0004, 0.01, size=(len(idx), 4))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C", "D"])


def _check_weights(w: pd.Series) -> None:
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert (w >= -1e-8).all()


def test_mean_variance_weights() -> None:
    w = mean_variance_weights(sample_returns())
    _check_weights(w)


def test_risk_parity_weights() -> None:
    w = risk_parity_weights(sample_returns())
    _check_weights(w)


def test_max_diversification_weights() -> None:
    w = max_diversification_weights(sample_returns())
    _check_weights(w)


def test_black_litterman_weights() -> None:
    rets = sample_returns()
    assets = list(rets.columns)
    market_weights = pd.Series(np.ones(len(assets)) / len(assets), index=assets)

    p = pd.DataFrame([[1.0, -1.0, 0.0, 0.0]], columns=assets, index=["v1"])
    q = pd.Series([0.01], index=p.index)
    conf = pd.Series([0.7], index=p.index)

    w = black_litterman_weights(rets, market_weights=market_weights, p=p, q=q, confidence=conf)
    _check_weights(w)
