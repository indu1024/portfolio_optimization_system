from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices_from_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError("Price file must include a Date column")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], utc=False)
    out = out.dropna(subset=["Date"])
    out = out.sort_values("Date")
    out = out.drop_duplicates(subset=["Date"], keep="last")

    asset_cols = [c for c in out.columns if c != "Date"]
    if len(asset_cols) < 2:
        raise ValueError("Provide at least 2 asset columns in the price file")

    out = out.set_index("Date")[asset_cols].astype(float)
    out = out.dropna(how="all")
    out = out.ffill().dropna()
    return out


def returns_from_prices(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    if method not in {"log", "simple"}:
        raise ValueError("method must be one of: log, simple")

    if method == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna()


def load_market_weights(market_weights_csv: str | Path, asset_columns: list[str]) -> pd.Series:
    df = pd.read_csv(market_weights_csv)
    required = {"asset", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Market weights file missing columns: {sorted(missing)}")

    series = df.set_index("asset")["weight"].astype(float)
    series = series.reindex(asset_columns).fillna(0.0)
    total = float(series.sum())
    if total <= 0:
        raise ValueError("Market weights must sum to a positive number")
    return series / total


def load_bl_views(views_csv: str | Path, asset_columns: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Expected format (one row per asset-loading in each view):
      view,asset,loading,view_return,confidence
      v1,AssetA,1,0.02,0.6
      v1,AssetB,-1,0.02,0.6
      v2,AssetC,1,0.01,0.8
      v2,AssetD,-1,0.01,0.8
    """
    df = pd.read_csv(views_csv)
    required = {"view", "asset", "loading", "view_return", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"BL views file missing columns: {sorted(missing)}")

    rows: list[pd.Series] = []
    q_vals: list[float] = []
    conf_vals: list[float] = []

    for view_name, grp in df.groupby("view", sort=True):
        row = pd.Series(0.0, index=asset_columns)
        for _, r in grp.iterrows():
            asset = r["asset"]
            if asset not in row.index:
                raise ValueError(f"Unknown asset in view {view_name}: {asset}")
            row.loc[asset] = float(r["loading"])

        q = float(grp["view_return"].iloc[0])
        confidence = float(grp["confidence"].iloc[0])
        if not 0 < confidence <= 1:
            raise ValueError(f"confidence must be in (0,1] for view {view_name}")

        rows.append(row)
        q_vals.append(q)
        conf_vals.append(confidence)

    if not rows:
        raise ValueError("No views found in views file")

    p = pd.DataFrame(rows)
    p.index = sorted(df["view"].unique())
    q = pd.Series(q_vals, index=p.index)
    conf = pd.Series(conf_vals, index=p.index)
    return p, q, conf
