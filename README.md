# Portfolio Optimization System (Python)

This project compares multiple portfolio construction methods on the same asset universe using rolling out-of-sample (OOS) evaluation.

Implemented methods:
- Mean-variance optimization (utility maximization)
- Risk parity (equal risk contribution)
- Maximum diversification (maximize diversification ratio)
- Black-Litterman (market equilibrium + investor views)
- Equal-weight benchmark

## What it outputs
- OOS daily return series by strategy
- OOS performance metrics (annual return/volatility, Sharpe, max drawdown, Calmar)
- Rebalance-date weights by strategy
- Cumulative return plot
- Markdown report with strategy ranking

## Project structure

```
portfolio_optimization_system/
  data/
    multi_asset_prices_sample.csv
    market_weights_sample.csv
    bl_views_sample.csv
  src/portfolio_optimization_system/
    cli.py
    data_loader.py
    optimizers.py
    backtest.py
    report.py
  tests/
    test_optimizers.py
    test_backtest.py
  pyproject.toml
  README.md
```

## Install

```bash
cd /Users/indumathi/Documents/Playground/portfolio_optimization_system
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Input formats

### Prices CSV (required)
Wide format with Date + asset columns:

```csv
Date,US_EQ,EU_EQ,EM_EQ,US_BOND,GOLD
2021-01-04,100.1,99.8,102.3,100.0,100.5
2021-01-05,100.8,100.2,101.6,100.1,99.9
```

### Market weights CSV (optional, BL)

```csv
asset,weight
US_EQ,0.35
EU_EQ,0.20
EM_EQ,0.15
US_BOND,0.20
GOLD,0.10
```

### BL views CSV (optional)
One row per asset loading per view:

```csv
view,asset,loading,view_return,confidence
v1,US_EQ,1,0.02,0.7
v1,EU_EQ,-1,0.02,0.7
v2,GOLD,1,0.01,0.6
v2,US_BOND,-1,0.01,0.6
```

## Run

```bash
portfolio-opt \
  --prices-csv /Users/indumathi/Documents/Playground/portfolio_optimization_system/data/multi_asset_prices_sample.csv \
  --market-weights-csv /Users/indumathi/Documents/Playground/portfolio_optimization_system/data/market_weights_sample.csv \
  --views-csv /Users/indumathi/Documents/Playground/portfolio_optimization_system/data/bl_views_sample.csv \
  --train-window 252 \
  --rebalance-step 21 \
  --risk-aversion 3.0 \
  --bl-tau 0.05 \
  --symbol-group sample_universe \
  --output-dir /Users/indumathi/Documents/Playground/portfolio_optimization_system/outputs
```

If you omit `--market-weights-csv` and `--views-csv`, the code uses equal market weights and a conservative default view.

## Why this is useful
You can evaluate optimizer robustness by comparing OOS behavior, not just in-sample fit. That makes strategy selection more realistic for deployment.
