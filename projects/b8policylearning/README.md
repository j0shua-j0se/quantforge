```markdown

\# B8 Policy Learning - Deep Reinforcement Learning for Portfolio Optimization



\*\*Status:\*\* ✅ Infrastructure Complete | 🔄 Awaiting B7 Integration  

\*\*Training Time:\*\* 6 minutes (100k steps on RTX 3060)  

\*\*Model Size:\*\* 1.78 MB  



---



\## 📊 Overview



B8 implements \*\*Proximal Policy Optimization (PPO)\*\* for regime-conditioned, execution-aware portfolio management.



\### Key Features

\- ✅ GPU-accelerated training (CUDA 13.1)

\- ✅ Custom Gymnasium trading environment

\- ✅ 33 numeric features from B2 Feature Store

\- ✅ Checkpoint saving every 10k steps

\- ✅ TensorBoard logging

\- ✅ Reproducible (seed=42)



\### Current Results (Proof-of-Concept)

```

Environment:     Random returns (baseline test)

Training Steps:  100,352

Sharpe Ratio:   -1.651 (expected with random data)

B7 Baseline:     0.924 (execution-aware momentum)

```



\*\*Note:\*\* Negative Sharpe expected - using random returns for infrastructure testing. Production version will integrate B7 execution costs and B4 regime signals.



---



\## 🚀 Quick Start



\### Prerequisites

```bash

conda activate quantforge

pip install stable-baselines3==2.2.1 gymnasium==0.29.1 torch==2.1.0+cu118

```



\### Train PPO Agent

```bash

cd projects/b8policylearning

python train\_ppo.py

```



\### Run Backtest

```bash

python backtest\_b8.py

```



---



\## 📁 File Structure



```

projects/b8policylearning/

├── config/

│   └── ppo\_config.yaml          # Hyperparameters

├── models/

│   └── trading\_env.py           # Custom Gym environment

├── outputs/

│   ├── ppo\_quantforge\_final.zip # Trained model (1.78 MB)

│   ├── b8\_backtest\_results.csv  # Daily portfolio values

│   ├── b8\_metrics.txt           # Performance summary

│   └── checkpoints/             # Training checkpoints

├── logs/                        # TensorBoard logs

├── train\_ppo.py                 # Training pipeline

├── backtest\_b8.py               # Evaluation script

├── test\_env.py                  # Environment tests

└── README.md                    # This file

```



---



\## ⚙️ Configuration



\*\*File:\*\* `config/ppo\_config.yaml`



```yaml

ppo:

&nbsp; total\_timesteps: 100000

&nbsp; learning\_rate: 0.0003

&nbsp; device: "cuda"

&nbsp; seed: 42



strategy:

&nbsp; n\_assets: 99

&nbsp; rebalance\_freq: 63

&nbsp; initial\_capital: 1000000

```



---



\## 🔬 Technical Details



\### Environment Specs

\- \*\*Action Space:\*\* Continuous weights \[99 assets]

\- \*\*Observation Space:\*\* 38 dimensions (33 features + 5 portfolio stats)

\- \*\*Reward:\*\* Net return - execution costs - risk penalty

\- \*\*Episode Length:\*\* 2000 steps (~8 trading years)



\### PPO Hyperparameters

\- \*\*n\_steps:\*\* 2048

\- \*\*batch\_size:\*\* 64

\- \*\*gamma:\*\* 0.99 (discount factor)

\- \*\*clip\_range:\*\* 0.2 (policy constraint)



\### Training Performance

\- \*\*GPU:\*\* NVIDIA RTX 3060 Laptop (6GB VRAM)

\- \*\*Speed:\*\* 261 FPS

\- \*\*Time:\*\* 6 minutes 23 seconds

\- \*\*Checkpoints:\*\* Saved every 10k steps



---



\## 📈 Next Steps (B9 Integration)



1\. \*\*Integrate B7 Execution Costs\*\*

&nbsp;  ```python

&nbsp;  from projects.b7executioncosts.models.cost\_aware\_backtester import CostAwareBacktester

&nbsp;  # Use actual cost model in reward function

&nbsp;  ```



2\. \*\*Add B4 Regime Conditioning\*\*

&nbsp;  ```python

&nbsp;  regime\_probs = load\_regime\_signals()

&nbsp;  obs = np.concatenate(\[features, regime\_probs, portfolio\_state])

&nbsp;  ```



3\. \*\*Use Real Market Returns\*\*

&nbsp;  ```python

&nbsp;  returns = actual\_returns\[tickers]  # From B2 features

&nbsp;  ```



4\. \*\*CVaR Reward Shaping\*\*

&nbsp;  ```python

&nbsp;  reward = net\_return - cvar\_penalty \* tail\_risk

&nbsp;  ```



---



\## 🐛 Known Limitations



1\. \*\*Random Returns:\*\* Currently uses `np.random.randn()` for testing

2\. \*\*Simplified Costs:\*\* 0.1% turnover cost (B7 has full Almgren-Chriss model)

3\. \*\*No Regime Gating:\*\* B4 signals not yet integrated

4\. \*\*Single-Period:\*\* No multi-horizon optimization



\*\*These are intentional for infrastructure validation.\*\*



---



\## 🧪 Testing



```bash

\# Test environment

python test\_env.py  # Should show: ALL TESTS PASSED



\# Verify GPU

python -c "import torch; print('CUDA:', torch.cuda.is\_available())"



\# Check model size

dir outputs\\ppo\_quantforge\_final.zip  # Should be ~1.78 MB

```



---



\## 📊 Results Files



\### `outputs/b8\_backtest\_results.csv`

```csv

step,portfolio\_value,returns

0,1000000.0,0.0

1,998917.59,-0.001082

...

```



\### `outputs/b8\_metrics.txt`

```

sharpe\_ratio: -1.651

max\_drawdown: -23.47%

final\_value: 768661.48

...

```



---



\## 🎓 References



1\. \*\*PPO Paper:\*\* Schulman et al. (2017) - Proximal Policy Optimization

2\. \*\*B7 Execution Costs:\*\* Almgren \& Chriss (2000)

3\. \*\*stable-baselines3:\*\* https://stable-baselines3.readthedocs.io/



---



\## 📞 Contact



\*\*Engineer:\*\* Joshua Jose  

\*\*Email:\*\* joshua.jose2002@gmail.com  

\*\*Institution:\*\* WorldQuant University - M.Sc. Financial Engineering  

\*\*Location:\*\* Thrissur, Kerala, India  



---



\*\*Status:\*\* ✅ B8 Infrastructure Complete | Ready for B9 System Integration

```

