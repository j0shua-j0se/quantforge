```markdown

\# B7: Execution Cost Modeling 💰



\*\*Realistic trading costs for portfolio backtesting\*\*



---



\## 📊 Results Summary



| Metric | Value |

|--------|-------|

| \*\*Gross Sharpe Ratio\*\* | 0.932 |

| \*\*Net Sharpe Ratio\*\* | 0.924 |

| \*\*Cost Impact\*\* | 0.9% degradation |

| \*\*Total Costs (10 years)\*\* | $46,264 |

| \*\*Average Cost per Rebalance\*\* | $1,157 |



✅ \*\*Conclusion:\*\* Costs have minimal impact (<1% Sharpe degradation). Strategy remains profitable after realistic execution costs.



---



\## 🎯 What B7 Does



B7 adds \*\*three types of execution costs\*\* to backtests:



1\. \*\*Bid-Ask Spread\*\* (2-10 bps based on market cap)

2\. \*\*Market Impact\*\* (Almgren-Chriss square-root law)

3\. \*\*Transaction Fees\*\* (Interactive Brokers pricing: $1 + $0.005/share)



---



\## 🚀 How to Run



```cmd

cd projects\\b7executioncosts

python pipeline.py

```



\*\*Output:\*\*

\- `outputs/b7\_backtest\_results.csv` - Daily returns with costs

\- `outputs/b7\_metrics.txt` - Performance summary



---



\## 📁 Project Structure



```

b7executioncosts/

├── config/

│   └── execution\_config.yaml     # Cost parameters

├── models/

│   ├── execution\_model.py        # Cost calculator

│   └── cost\_aware\_backtester.py  # Backtester

├── outputs/

│   ├── b7\_backtest\_results.csv

│   └── b7\_metrics.txt

├── logs/

└── pipeline.py                    # Main script

```



---



\## 📖 Cost Model Details



\### Spread Costs

\- Large cap (>$100B): 2 bps

\- Mid cap ($10B-$100B): 5 bps

\- Small cap (<$10B): 10 bps



\### Market Impact (Square-Root Law)

```

Impact = k \* (trade\_size / daily\_volume)^0.6 \* volatility

```

Where k = 0.7 (calibrated to institutional data)



\### Transaction Fees

```

Fee = max($1, $0.005 \* shares)

Capped at 1% of trade value

```



---



\## 🎓 Built for QuantForge Thesis



\*\*Author:\*\* Joshua Jose  

\*\*Date:\*\* February 12, 2026  

\*\*Phase:\*\* B7 of 11 (Complete ✅)  

\*\*Institution:\*\* WorldQuant University



---



\## 📚 References



1\. Almgren \& Chriss (2000) - Optimal execution of portfolio transactions

2\. Grinold \& Kahn (2000) - Active Portfolio Management



---



\*\*Next Phase:\*\* B8 - Policy Learning with Reinforcement Learning

```

