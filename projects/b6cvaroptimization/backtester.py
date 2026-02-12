"""B6 Walk-Forward Backtester"""
import pandas as pd
import numpy as np
from typing import Dict
from optimizer import CVaROptimizer

class WalkForwardBacktester:
    def __init__(self, optimizer, train_window=756, test_window=252):
        self.optimizer = optimizer
        self.train_window = train_window
        self.test_window = test_window
        print(f"Backtester initialized (train={train_window}d, test={test_window}d)")
    
    def backtest(self, features, signals, regimes, config):
        print("="*60)
        print("STARTING WALK-FORWARD BACKTEST")
        print("="*60)
        
        investable = signals[signals['signal'].isin(['STRONG_BUY', 'BUY'])]
        tickers = investable['ticker'].tolist()
        print(f"Investable universe: {len(tickers)} stocks")
        
        returns_panel = features[features['ticker'].isin(tickers)].pivot_table(
            index='date', columns='ticker', values='returns'
        ).fillna(0)
        
        returns_panel.index = pd.to_datetime(returns_panel.index)
        returns_panel = returns_panel.sort_index()
        
        print(f"Returns panel: {returns_panel.shape[0]} days x {returns_panel.shape[1]} assets")
        
        risk_scores = investable.set_index('ticker')['risk_score']
        
        results = []
        n_samples = len(returns_panel)
        start_idx = self.train_window
        
        cycle = 0
        while start_idx + self.test_window <= n_samples:
            cycle += 1
            
            train_start = start_idx - self.train_window
            train_end = start_idx
            test_start = start_idx
            test_end = start_idx + self.test_window
            
            train_dates = returns_panel.index[train_start:train_end]
            test_dates = returns_panel.index[test_start:test_end]
            
            print(f"\nCycle {cycle}")
            print(f"Train: {train_dates[0].date()} to {train_dates[-1].date()}")
            print(f"Test:  {test_dates[0].date()} to {test_dates[-1].date()}")
            
            train_returns = returns_panel.iloc[train_start:train_end]
            current_regime = self._get_regime(train_dates[-1], regimes)
            
            weights, opt_results = self.optimizer.optimize(
                returns_df=train_returns.T,
                risk_scores=risk_scores,
                regime=current_regime,
                regime_constraints=config['optimizer']['regime_constraints']
            )
            
            test_returns = returns_panel.iloc[test_start:test_end]
            portfolio_returns = (test_returns * weights).sum(axis=1)
            
            perf = self._compute_performance(portfolio_returns)
            
            results.append({
                'cycle': cycle,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'regime': current_regime,
                'n_assets': opt_results.get('n_assets', len(tickers)),
                **perf
            })
            
            print(f"Performance: Return={perf['total_return']:.2%} | Sharpe={perf['sharpe']:.2f} | MaxDD={perf['max_drawdown']:.2%}")
            
            start_idx += self.test_window
        
        print("="*60)
        print(f"BACKTEST COMPLETE ({cycle} cycles)")
        print("="*60)
        
        return pd.DataFrame(results)

    def _get_regime(self, date, regimes):
        regimes['date'] = pd.to_datetime(regimes['date']).dt.tz_localize(None)
        date = pd.Timestamp(date).tz_localize(None)
        regimes = regimes.sort_values('date')
        past_regimes = regimes[regimes['date'] <= date]
        if len(past_regimes) == 0:
            return 1
        return int(past_regimes.iloc[-1]['regime'])

    
    def _compute_performance(self, returns):
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / (volatility + 1e-9)
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'cvar_95': cvar_95
        }
