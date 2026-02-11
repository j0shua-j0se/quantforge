Perfect! Let's add the professional README! 📝



```markdown

\# B3 Baseline Strategies



\*\*Complete momentum strategy implementation with walk-forward safe backtesting.\*\*



\## 🎯 Overview



B3 implements baseline trading strategies with realistic transaction costs and professional backtesting infrastructure.



\## ✅ Features



\- \*\*Momentum Strategy\*\*: Long-short strategy buying winners, shorting losers

\- \*\*Transaction Costs\*\*: Commission (0.05%) + Slippage (0.03%)

\- \*\*Walk-Forward Safe\*\*: No lookahead bias

\- \*\*Full Pipeline\*\*: Config → Signals → Backtest → Results



\## 📊 Results



\*\*7-Year Backtest (2018-2024)\*\*

\- Tested on: 174,240 records (99 tickers)

\- Trading days: 1,760

\- Signals: 52,800 longs, 51,040 shorts



\## 🚀 Quick Start



```bash

\# Activate environment

conda activate quantforge



\# Run full pipeline

python projects/b3\_baselines/pipeline.py



\# Results saved to outputs/

```



\## 📁 Structure



```

b3\_baselines/

├── README.md              # This file

├── config.yaml            # Strategy configuration

├── pipeline.py            # Main orchestrator

├── backtest.py            # Backtesting engine

├── strategies/

│   ├── \_\_init\_\_.py

│   └── momentum.py        # Momentum strategy

└── outputs/               # Results (CSV, metrics)

```



\## 🔧 Configuration



Edit `config.yaml` to change:

\- Strategy parameters (long/short percentages)

\- Starting capital ($1M default)

\- Transaction costs

\- Rebalancing frequency



\## 🧪 Testing



```bash

\# Test momentum strategy only

python projects/b3\_baselines/strategies/momentum.py



\# Test backtester only

python projects/b3\_baselines/backtest.py

```



\## 📈 Output Files



After running `pipeline.py`, check `outputs/` for:

\- `equity\_curve\_YYYYMMDD\_HHMMSS.csv` - Daily portfolio values

\- `metrics\_YYYYMMDD\_HHMMSS.txt` - Performance summary

\- `signals\_sample\_YYYYMMDD\_HHMMSS.csv` - Trading signals sample



\## 📐 Strategy Details



\*\*Momentum Strategy\*\*

\- Uses `momentum\_60d` feature from B2

\- Long top 30% performers

\- Short bottom 30% performers

\- Equal-weight positions

\- Daily rebalancing



\*\*Transaction Costs\*\*

\- Commission: 0.05% per trade

\- Slippage: 0.03% per trade

\- Total: ~0.08% round-trip cost



\## 🔬 Walk-Forward Safety



All features are lagged by 1 day to prevent lookahead bias:

\- Decision at time `t` uses only data from time `t-1`

\- No future information leakage

\- Realistic out-of-sample testing



\## 📝 Notes



This is a \*\*research platform\*\* showing honest performance including all costs. Strategy performance varies by market regime. The implementation prioritizes:

1\. \*\*Correctness\*\* over performance

2\. \*\*Reproducibility\*\* over complexity

3\. \*\*Transparency\*\* over optimization



\## 🚀 Future Enhancements



\- \[ ] Add mean-reversion strategy

\- \[ ] Add SMA crossover strategy

\- \[ ] Implement walk-forward validation windows

\- \[ ] Add performance visualization (matplotlib)

\- \[ ] Risk management (stop-loss, position sizing)

\- \[ ] Multi-strategy portfolio



\## 🔗 Links



\- \*\*Project\*\*: \[QuantForge](https://github.com/j0shua-j0se/quantforge)

\- \*\*Documentation\*\*: See main README.md

\- \*\*Dependencies\*\*: B0 (Infrastructure), B1 (Data), B2 (Features)



---



\*\*Built with ❤️ as part of QuantForge - Execution-Aware Portfolio Policy Learning\*\*



\*\*Date\*\*: February 12, 2026  

\*\*Version\*\*: v0.4.0-b3-complete  

\*\*Status\*\*: ✅ Production Ready

```

