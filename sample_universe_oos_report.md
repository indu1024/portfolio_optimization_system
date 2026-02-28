# Portfolio Optimization OOS Report (sample_universe)

## Ranking (by Sharpe)
- risk_parity: sharpe=0.889, ann_return=7.58%, ann_vol=8.53%, max_dd=-13.02%
- equal_weight: sharpe=0.889, ann_return=7.58%, ann_vol=8.53%, max_dd=-13.02%
- black_litterman: sharpe=0.831, ann_return=7.48%, ann_vol=9.00%, max_dd=-14.12%
- max_diversification: sharpe=0.830, ann_return=5.67%, ann_vol=6.83%, max_dd=-10.60%
- mean_variance: sharpe=0.594, ann_return=11.46%, ann_vol=19.30%, max_dd=-31.70%

## Key Takeaway
- Best out-of-sample Sharpe strategy: `risk_parity`
- Compare this with max drawdown and total return before selecting a production allocation rule.

## Plot
- Cumulative returns: `sample_universe_oos_cumulative_returns.png`