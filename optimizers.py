from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def _normalize_weights(w: np.ndarray, long_only: bool = True) -> np.ndarray:
    x = np.asarray(w, dtype=float)
    if long_only:
        x = np.maximum(x, 0.0)
    total = x.sum()
    if total <= 0:
        return np.ones_like(x) / len(x)
    return x / total


def _annualized_cov(returns: pd.DataFrame, trading_days: int = TRADING_DAYS) -> np.ndarray:
    return returns.cov().values * trading_days


def _annualized_mu(returns: pd.DataFrame, trading_days: int = TRADING_DAYS) -> np.ndarray:
    return returns.mean().values * trading_days


def mean_variance_weights(
    returns: pd.DataFrame,
    risk_aversion: float = 3.0,
    long_only: bool = True,
) -> pd.Series:
    cols = list(returns.columns)
    n = len(cols)
    mu = _annualized_mu(returns)
    cov = _annualized_cov(returns)
    cov = cov + np.eye(n) * 1e-8

    def objective(w: np.ndarray) -> float:
        ret = mu @ w
        var = w @ cov @ w
        return -(ret - 0.5 * risk_aversion * var)

    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        w = x0
    else:
        w = res.x
    w = _normalize_weights(w, long_only=long_only)
    return pd.Series(w, index=cols, name="mean_variance")


def risk_parity_weights(returns: pd.DataFrame, long_only: bool = True) -> pd.Series:
    cols = list(returns.columns)
    n = len(cols)
    cov = _annualized_cov(returns)
    cov = cov + np.eye(n) * 1e-8

    def objective(w: np.ndarray) -> float:
        w = _normalize_weights(w, long_only=long_only)
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return 1e6
        mrc = cov @ w
        rc = w * mrc
        target = np.full(n, port_var / n)
        return float(np.sum((rc - target) ** 2))

    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = res.x if res.success else x0
    w = _normalize_weights(w, long_only=long_only)
    return pd.Series(w, index=cols, name="risk_parity")


def max_diversification_weights(returns: pd.DataFrame, long_only: bool = True) -> pd.Series:
    cols = list(returns.columns)
    n = len(cols)
    cov = _annualized_cov(returns)
    cov = cov + np.eye(n) * 1e-8
    vols = np.sqrt(np.diag(cov))

    def objective(w: np.ndarray) -> float:
        w = _normalize_weights(w, long_only=long_only)
        weighted_vol = float(w @ vols)
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol <= 0:
            return 1e6
        div_ratio = weighted_vol / port_vol
        return -div_ratio

    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = res.x if res.success else x0
    w = _normalize_weights(w, long_only=long_only)
    return pd.Series(w, index=cols, name="max_diversification")


def black_litterman_posterior(
    returns: pd.DataFrame,
    market_weights: pd.Series,
    p: pd.DataFrame,
    q: pd.Series,
    confidence: pd.Series,
    tau: float = 0.05,
    risk_aversion: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    cols = list(returns.columns)
    cov = _annualized_cov(returns)
    cov = cov + np.eye(len(cols)) * 1e-8

    w_mkt = market_weights.reindex(cols).fillna(0.0).values
    if w_mkt.sum() <= 0:
        w_mkt = np.ones(len(cols)) / len(cols)
    else:
        w_mkt = w_mkt / w_mkt.sum()

    pi = risk_aversion * cov @ w_mkt

    p_mat = p.reindex(columns=cols).values
    q_vec = q.values

    omega_diag = np.diag(np.diag(p_mat @ (tau * cov) @ p_mat.T))
    omega = omega_diag / confidence.values[:, None]
    omega = (omega + omega.T) / 2.0

    tau_cov_inv = np.linalg.pinv(tau * cov)
    omega_inv = np.linalg.pinv(omega)

    middle = np.linalg.pinv(tau_cov_inv + p_mat.T @ omega_inv @ p_mat)
    posterior_mean = middle @ (tau_cov_inv @ pi + p_mat.T @ omega_inv @ q_vec)
    posterior_cov = cov + middle
    return posterior_mean, posterior_cov


def black_litterman_weights(
    returns: pd.DataFrame,
    market_weights: pd.Series,
    p: pd.DataFrame,
    q: pd.Series,
    confidence: pd.Series,
    tau: float = 0.05,
    risk_aversion: float = 3.0,
    long_only: bool = True,
) -> pd.Series:
    cols = list(returns.columns)
    n = len(cols)

    mu_bl, cov_bl = black_litterman_posterior(
        returns=returns,
        market_weights=market_weights,
        p=p,
        q=q,
        confidence=confidence,
        tau=tau,
        risk_aversion=risk_aversion,
    )

    def objective(w: np.ndarray) -> float:
        ret = mu_bl @ w
        var = w @ cov_bl @ w
        return -(ret - 0.5 * risk_aversion * var)

    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)

    w = res.x if res.success else x0
    w = _normalize_weights(w, long_only=long_only)
    return pd.Series(w, index=cols, name="black_litterman")
