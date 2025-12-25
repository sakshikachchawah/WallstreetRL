# -*- coding: utf-8 -*-
"""Untitled9.ipynb - Corrected Version"""

import os
import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

import yfinance as yf
try:
    import gym
    from gym import spaces
except Exception:
    gym = None
    spaces = None
    print('gym not available. If running locally install with: pip install gym==0.26.2')
import ta

np.set_printoptions(precision=4, suppress=True)
plt.style.use('seaborn-v0_8')

# ============== DATA.PY ==============

def fetch_data(ticker='AAPL', start='2019-01-01', end=None, interval='1d'):
    """Fetch historical OHLCV data using yfinance."""
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
    df = df.rename(columns=str.lower)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

def add_features(df):
    """Feature engineering with technical indicators."""
    df = df.copy()
    df['return'] = df['close'].pct_change()
    
    # RSI
    df['rsi_14'] = ta.momentum.rsi(df['close'].squeeze(), window=14)
    
    # MACD
    macd = ta.trend.MACD(close=df['close'].squeeze(), window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['close'].squeeze(), window=20, window_dev=2)
    df['bb_h'] = bb.bollinger_hband().squeeze()
    df['bb_l'] = bb.bollinger_lband().squeeze()
    
    bb_h_arr = df['bb_h'].values
    bb_l_arr = df['bb_l'].values
    close_arr = df['close'].values
    
    if bb_h_arr.ndim > 1:
        bb_h_arr = bb_h_arr.flatten()
    if bb_l_arr.ndim > 1:
        bb_l_arr = bb_l_arr.flatten()
    if close_arr.ndim > 1:
        close_arr = close_arr.flatten()
    
    df['bb_w'] = pd.Series((bb_h_arr - bb_l_arr) / close_arr, index=df.index)
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    df = df.dropna()
    return df

def normalize_features(df, feature_cols):
    """Normalize features with rolling z-score."""
    df = df.copy()
    for col in feature_cols:
        roll_mean = df[col].rolling(200).mean()
        roll_std = df[col].rolling(200).std().replace(0, np.nan)
        # Changed: use _norm_ (with trailing underscore) for consistency
        df[col + '_norm_'] = (df[col] - roll_mean) / roll_std
    df = df.dropna()
    return df

def train_test_split_df(df, test_size=0.2):
    n = len(df)
    test_n = int(n * test_size)
    train_df = df.iloc[:-test_n]
    test_df = df.iloc[-test_n:]
    return train_df, test_df

def prepare_dataset(ticker='AAPL', start='2019-01-01', end=None):
    """Prepare and split dataset with proper column handling."""
    df = fetch_data(ticker=ticker, start=start, end=end)
    df = add_features(df)
    
    # Store original close prices
    original_close_series = df['close'].copy()
    
    feature_cols = [
        'close','return','rsi_14','macd','macd_signal','macd_hist',
        'bb_h','bb_l','bb_w','sma_20','sma_50'
    ]
    df = normalize_features(df, feature_cols)
    
    # Flatten column names
    df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]
    
    # Select normalized columns (now ending with _norm_)
    norm_cols_df = df.filter(like='_norm_')
    
    # Align and add original close column
    aligned_original_close_series = original_close_series.reindex(norm_cols_df.index)
    
    final_df = norm_cols_df.copy()
    final_df['close'] = aligned_original_close_series
    final_df = final_df.dropna()
    
    print(f"Columns in final_df: {final_df.columns.tolist()}")
    
    train_df, test_df = train_test_split_df(final_df, test_size=0.2)
    return train_df, test_df

# ============== ENV.PY ==============

class TradingEnv(gym.Env):
    """Gym-compatible trading environment."""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, feature_cols, initial_balance=10000.0, transaction_cost=0.001, max_shares=None):
        super().__init__()
        self.df = df.copy()
        
        # Ensure 'close' column exists
        if 'close' not in self.df.columns:
            raise ValueError(f"'close' column not found in df. Available columns: {self.df.columns.tolist()}")
        
        self.prices = self.df['close'].values
        self.features = self.df[feature_cols].values
        self.feature_cols = feature_cols
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_shares = max_shares
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(feature_cols) + 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        self.reset()
    
    def _get_obs(self):
        feat = self.features[self.t].astype(np.float32)
        price = self.prices[self.t]
        balance_norm = np.float32(self.balance / (self.initial_balance + 1e-8))
        shares_norm = np.float32(self.shares_held / (self.max_shares if self.max_shares else 1 + 1e-8))
        pos_flag = np.float32(1.0 if self.shares_held > 0 else 0.0)
        obs = np.concatenate([feat, np.array([balance_norm, pos_flag, shares_norm], dtype=np.float32)], axis=0)
        return obs
    
    def _portfolio_value(self, price=None):
        if price is None:
            price = self.prices[self.t]
        return self.balance + self.shares_held * price
    
    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        info = {}
        
        price = self.prices[self.t]
        prev_value = self._portfolio_value(price)
        cost_multiplier = 1.0 + self.transaction_cost
        
        if action == 1:  # Buy
            if self.shares_held == 0:
                max_shares = self.max_shares or int(self.balance / (price * cost_multiplier))
                buy_shares = max_shares
                if buy_shares > 0:
                    spend = buy_shares * price * cost_multiplier
                    self.balance -= spend
                    self.shares_held += buy_shares
            else:
                info['penalty'] = -0.001
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * price * (1.0 - self.transaction_cost)
                self.balance += revenue
                self.shares_held = 0
            else:
                info['penalty'] = -0.001
        
        self.t += 1
        if self.t >= self.n_steps - 1:
            done = True
        
        new_price = self.prices[self.t]
        curr_value = self._portfolio_value(new_price)
        reward = curr_value - prev_value
        if 'penalty' in info:
            reward += info['penalty']
        
        obs = self._get_obs()
        return obs, np.float32(reward), done, info
    
    def reset(self, seed=None, options=None):
        self.t = 0
        self.balance = float(self.initial_balance)
        self.shares_held = 0
        if self.max_shares is None and self.prices.size > 0:
            first_price = self.prices[0]
            self.max_shares_runtime = int(self.initial_balance / (first_price * (1.0 + self.transaction_cost)))
        else:
            self.max_shares_runtime = self.max_shares
        return self._get_obs()
    
    def render(self):
        price = self.prices[self.t]
        print(f"t={self.t} price={price:.2f} balance={self.balance:.2f} shares={self.shares_held} value={self._portfolio_value(price):.2f}")

# ============== AGENT.PY ==============

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128, 64)):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 buffer_capacity=100000, batch_size=64, target_update_every=200, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_every = target_update_every
        self.learn_step = 0
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(capacity=buffer_capacity)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            action = torch.argmax(q_values, dim=1).item()
            return action
    
    def push(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        self.learn_step += 1
        if self.learn_step % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()
    
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
    
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))

# ============== TRAIN.PY ==============

def run_training(train_df, episodes=20, initial_balance=10000.0, transaction_cost=0.001, model_path='dqn_wallstreetrl.pt'):
    feature_cols = [c for c in train_df.columns if c.endswith('_norm_')]
    env = TradingEnv(train_df, feature_cols, initial_balance=initial_balance, transaction_cost=transaction_cost)
    
    state_dim = len(feature_cols) + 3
    action_dim = 3
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.995, epsilon_start=1.0, epsilon_end=0.05,
                     epsilon_decay=0.995, buffer_capacity=200000, batch_size=128, target_update_every=500)
    
    episode_rewards = []
    episode_values = []
    best_avg_reward = -np.inf
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        values = []
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward, next_state, float(done))
            loss = agent.update()
            
            state = next_state
            total_reward += reward
            values.append(env._portfolio_value())
        
        episode_rewards.append(total_reward)
        episode_values.append(values)
        
        avg_last_5 = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else np.mean(episode_rewards)
        if avg_last_5 > best_avg_reward:
            best_avg_reward = avg_last_5
            agent.save(model_path)
        
        print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    plt.figure(figsize=(8,4))
    plt.plot(episode_rewards, label='Reward per episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training reward')
    plt.legend()
    plt.show()
    
    return agent, feature_cols, episode_rewards, episode_values

# ============== EVAL.PY ==============

def simulate_agent(env_df, feature_cols, agent_path, initial_balance=10000.0, transaction_cost=0.001):
    env = TradingEnv(env_df, feature_cols, initial_balance=initial_balance, transaction_cost=transaction_cost)
    
    state_dim = len(feature_cols) + 3
    agent = DQNAgent(state_dim, 3)
    agent.load(agent_path)
    
    state = env.reset()
    done = False
    
    values = []
    actions = []
    prices = env_df['close'].values
    holdings = []
    balances = []
    
    while not done:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            q_values = agent.q_net(state_t)
            action = torch.argmax(q_values, dim=1).item()
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        values.append(env._portfolio_value())
        holdings.append(env.shares_held)
        balances.append(env.balance)
        state = next_state
    
    results = pd.DataFrame({
        'price': prices[:len(values)],
        'value': values,
        'action': actions[:len(values)],
        'shares': holdings,
        'balance': balances
    }, index=env_df.index[:len(values)])
    return results

def buy_and_hold(env_df, initial_balance=10000.0, transaction_cost=0.001):
    prices = env_df['close'].values
    first_price = prices[0]
    shares = int(initial_balance / (first_price * (1.0 + transaction_cost)))
    balance = initial_balance - shares * first_price * (1.0 + transaction_cost)
    values = shares * prices + balance
    actions = [1] + [0]*(len(prices)-1)
    return pd.DataFrame({'price': prices, 'value': values, 'action': actions}, index=env_df.index)

def sma_strategy(env_df, short=20, long=50, initial_balance=10000.0, transaction_cost=0.001):
    df = env_df.copy()
    df['sma_s'] = df['close'].rolling(short).mean()
    df['sma_l'] = df['close'].rolling(long).mean()
    df = df.dropna()
    prices = df['close'].values
    signals = (df['sma_s'] > df['sma_l']).astype(int)
    
    balance = initial_balance
    shares = 0
    values = []
    actions = []
    for i in range(len(prices)):
        price = prices[i]
        signal = signals.iloc[i]
        if signal == 1 and shares == 0:
            max_shares = int(balance / (price * (1.0 + transaction_cost)))
            spend = max_shares * price * (1.0 + transaction_cost)
            if max_shares > 0:
                shares += max_shares
                balance -= spend
            actions.append(1)
        elif signal == 0 and shares > 0:
            revenue = shares * price * (1.0 - transaction_cost)
            balance += revenue
            shares = 0
            actions.append(2)
        else:
            actions.append(0)
        values.append(balance + shares * price)
    
    res = pd.DataFrame({'price': prices, 'value': values, 'action': actions}, index=df.index)
    return res

def compute_metrics(values_series, rf_rate=0.0):
    v = values_series.values
    rets = pd.Series(v).pct_change().dropna()
    cum_return = (v[-1] / v[0]) - 1.0
    sharpe = (rets.mean() - rf_rate/252) / (rets.std() + 1e-8) * np.sqrt(252)
    roll_max = pd.Series(v).cummax()
    drawdown = pd.Series(v) / roll_max - 1.0
    max_dd = drawdown.min()
    total_profit = v[-1] - v[0]
    return {
        'Cumulative Return': cum_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Total Profit': total_profit,
    }

def count_trades(actions):
    actions = np.array(actions)
    buys = np.sum(actions == 1)
    sells = np.sum(actions == 2)
    return int(buys + sells)

# ============== MAIN ==============

if __name__ == "__main__":
    train_df, test_df = prepare_dataset(ticker='AAPL', start='2019-01-01')
    print(f"Train length: {len(train_df)} | Test length: {len(test_df)}")
    
    feature_cols_train = [c for c in train_df.columns if c.endswith('_norm_')]
    feature_cols_test = [c for c in test_df.columns if c.endswith('_norm_')]
    
    agent, feature_cols, episode_rewards, episode_values = run_training(
        train_df, episodes=25, initial_balance=10000.0, 
        transaction_cost=0.001, model_path='dqn_wallstreetrl.pt'
    )
    
    agent_results = simulate_agent(test_df, feature_cols, agent_path='dqn_wallstreetrl.pt')
    bah_results = buy_and_hold(test_df)
    sma_results = sma_strategy(test_df)
    
    metrics_rl = compute_metrics(agent_results['value'])
    metrics_bah = compute_metrics(bah_results['value'])
    metrics_sma = compute_metrics(sma_results['value'])
    
    num_trades_rl = count_trades(agent_results['action'])
    num_trades_bah = count_trades(bah_results['action'])
    num_trades_sma = count_trades(sma_results['action'])
    
    print("\nWallStreetRL (DQN) Test Metrics:")
    for k,v in metrics_rl.items():
        print(f" - {k}: {v:.4f}")
    print(f" - Number of Trades: {num_trades_rl}")
    
    print("\nBuy-and-Hold Test Metrics:")
    for k,v in metrics_bah.items():
        print(f" - {k}: {v:.4f}")
    print(f" - Number of Trades: {num_trades_bah}")
    
    print("\nSMA(20/50) Test Metrics:")
    for k,v in metrics_sma.items():
        print(f" - {k}: {v:.4f}")
    print(f" - Number of Trades: {num_trades_sma}")