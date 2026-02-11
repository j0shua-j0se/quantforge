```markdown

\# B2: Feature Store



\*\*Status:\*\* ✅ Complete  

\*\*Version:\*\* 0.3.0  

\*\*Date:\*\* February 12, 2026



---



\## Overview



Production-grade feature engineering pipeline that computes 25+ walk-forward safe features for quantitative trading strategies.



\## Features Generated



\### Price \& Momentum (15 features)

\- \*\*Returns\*\*: 1d, 5d, 20d, 60d (momentum)

\- \*\*Moving Averages\*\*: SMA (5, 10, 20, 50, 200 day)

\- \*\*Price Ratios\*\*: price/SMA for mean-reversion signals

\- \*\*Technical Indicators\*\*: RSI-14



\### Volatility (5 features)

\- \*\*Realized Volatility\*\*: 20d, 60d (annualized)

\- \*\*ATR\*\*: Average True Range (14-day)

\- \*\*Vol-of-Vol\*\*: Regime change detector



\### Volume (7 features)

\- \*\*Volume Averages\*\*: SMA-20, ratios

\- \*\*Dollar Volume\*\*: Liquidity proxy

\- \*\*OBV\*\*: On-Balance Volume (normalized)



\## Walk-Forward Safety ⚠️



\*\*Critical:\*\* All features are lagged by \*\*1 day\*\* to prevent lookahead bias.



```python

\# Feature at time t uses ONLY data from t-1 and earlier

df\['returns\_1d'] = df.groupby('ticker')\['close'].pct\_change().shift(1)

```



This ensures features are tradeable in real-time without future information.



\## Quick Start



```bash

\# Activate environment

conda activate quantforge



\# Run pipeline

python projects/b2\_features/pipeline.py



\# Output

data/features/features\_YYYYMMDD.parquet  # 62+ MB

data/features/manifest\_YYYYMMDD.json     # Feature metadata

```



\## Performance



\- \*\*Runtime:\*\* 2 seconds (248K records)

\- \*\*Features:\*\* 25 engineered features

\- \*\*Output Size:\*\* 62 MB (Parquet compressed)

\- \*\*Memory Usage:\*\* <500 MB peak



\## Configuration



Edit `configs/b2\_config.yaml` to customize:

\- Feature categories (enable/disable)

\- Window sizes for moving averages

\- Walk-forward lag amount

\- Quality thresholds



\## Output Format



\*\*Parquet file structure:\*\*

```python

{

&nbsp;   'date': datetime,

&nbsp;   'ticker': str,

&nbsp;   'close': float,

&nbsp;   # ... original OHLCV columns ...

&nbsp;   'returns\_1d': float,      # Lagged 1 day

&nbsp;   'sma\_20': float,          # 20-day SMA

&nbsp;   'realized\_vol\_20d': float,# Annualized volatility

&nbsp;   # ... 22 more features ...

}

```



\## Quality Checks



✅ Null percentage: <5% per feature  

✅ All features lagged for walk-forward safety  

✅ Statistics validated (mean, std, min, max)



\## Integration



Features ready for:

\- \*\*B3:\*\* Baseline strategies (momentum, mean-reversion, SMA crossover)

\- \*\*B6:\*\* CVaR portfolio optimization

\- \*\*B8:\*\* RL policy learning



\## Known Issues



\- `sma\_200` has 7.95% nulls → \*\*Expected\*\* (requires 200 days history)

\- FutureWarnings from pandas → \*\*Harmless\*\* (will be fixed in pandas 3.0)



\## Next Steps



1\. \*\*B3:\*\* Implement baseline strategies using these features

2\. \*\*Feature Expansion:\*\* Add technical indicators (Bollinger Bands, Stochastic)

3\. \*\*Cross-Sectional Features:\*\* Rank-based features across tickers



---



\*\*Author:\*\* Joshua Jose  

\*\*Project:\*\* QuantForge (FAU M.Sc. AI)  

\*\*Repository:\*\* https://github.com/j0shua-j0se/quantforge

```

