import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class DynamicFeatureEngineer:
    
    def __init__(self, num_assets, max_assets=100):
        self.num_assets = num_assets
        self.max_assets = max_assets
        self.obs_dim = max_assets + 14
        
    def get_observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
    
    def get_features(self, data, current_idx):
        features = []
        returns = data.pct_change()
        for ticker in data.columns:
            ret = returns[ticker].iloc[current_idx]
            features.append(ret if not np.isnan(ret) else 0.0)
        features.extend([0.0] * (self.max_assets - len(data.columns)))
        features.extend(self._calculate_volatility_features(data, current_idx))
        features.extend(self._calculate_correlation_features(data, current_idx))
        return np.array(features, dtype=np.float32)
    
    def _calculate_volatility_features(self, data, idx):
        returns = data.pct_change().iloc[:idx+1]
        if len(returns) < 5:
            return [0.0] * 10
        try:
            features = [
                returns.std(axis=1).iloc[-5:].mean() if len(returns) >= 5 else 0.0,
                returns.std(axis=1).iloc[-20:].mean() if len(returns) >= 20 else 0.0,
                returns.std(axis=1).iloc[-60:].mean() if len(returns) >= 60 else 0.0,
                returns.std().mean(),
                returns.std().std(),
                returns.skew().mean(),
                returns.kurtosis().mean(),
                (returns > 0).mean().mean(),
                returns.max(axis=1).iloc[-20:].mean() if len(returns) >= 20 else 0.0,
                returns.min(axis=1).iloc[-20:].mean() if len(returns) >= 20 else 0.0,
            ]
            return [f if np.isfinite(f) else 0.0 for f in features]
        except:
            return [0.0] * 10
    
    def _calculate_correlation_features(self, data, idx):
        returns = data.pct_change().iloc[max(0, idx-60):idx+1]
        if len(returns) < 20 or len(data.columns) < 2:
            return [0.0] * 4
        try:
            corr_matrix = returns.corr()
            np.fill_diagonal(corr_matrix.values, np.nan)
            features = [
                corr_matrix.mean().mean(),
                corr_matrix.std().std(),
                corr_matrix.max().max(),
                corr_matrix.min().min(),
            ]
            return [f if np.isfinite(f) else 0.0 for f in features]
        except:
            return [0.0] * 4


class DynamicPortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, tickers, data, initial_capital=100000, commission=0.001, 
                 max_position_size=0.15, min_position_size=0.01, max_assets=100):
        super().__init__()
        self.tickers = tickers
        self.num_assets = len(tickers)
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_assets = max_assets
        self.feature_engineer = DynamicFeatureEngineer(self.num_assets, self.max_assets)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = self.feature_engineer.get_observation_space()
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.holdings = np.zeros(self.num_assets)
        self.portfolio_history = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 60
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.num_assets)
        self.portfolio_history = []
        return self._get_observation(), {}
    
    def _get_observation(self):
        return self.feature_engineer.get_features(self.data, self.current_step)
    
    def step(self, action):
        action = action[:self.num_assets]
        action = np.abs(action)
        if action.sum() > 0:
            action = action / action.sum()
        else:
            action = np.ones(self.num_assets) / self.num_assets
        action = np.clip(action, self.min_position_size, self.max_position_size)
        action = action / action.sum()
        current_prices = self.data.iloc[self.current_step].values
        target_shares = (self.portfolio_value * action) / (current_prices + 1e-10)
        trades = target_shares - self.holdings
        commission_cost = np.abs(trades * current_prices).sum() * self.commission
        self.holdings = target_shares
        self.cash = self.portfolio_value - (self.holdings * current_prices).sum() - commission_cost
        self.current_step += 1
        if self.current_step < len(self.data):
            new_prices = self.data.iloc[self.current_step].values
            self.portfolio_value = self.cash + (self.holdings * new_prices).sum()
        reward = np.log(self.portfolio_value / self.portfolio_history[-1]) if self.portfolio_history else 0.0
        self.portfolio_history.append(self.portfolio_value)
        done = self.current_step >= self.max_steps - 1
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape[0])
        return obs, reward, done, False, {'portfolio_value': self.portfolio_value, 'step': self.current_step}
