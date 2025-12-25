# 📈 WallStreetRL

> Deep Reinforcement Learning for Algorithmic Trading

A sophisticated trading system that uses Deep Q-Networks (DQN) to learn optimal trading strategies from historical stock data. Compare the performance of an AI agent against traditional strategies like Buy & Hold and SMA crossovers.

## ✨ Features

- **🤖 Deep Q-Network Agent** - Learns to trade using reinforcement learning
- **📊 Interactive Dashboard** - Real-time visualization with Streamlit
- **⚖️ Strategy Comparison** - Benchmark against Buy & Hold and SMA strategies
- **📈 Technical Indicators** - RSI, MACD, Bollinger Bands, and moving averages
- **💰 Transaction Costs** - Realistic trading simulation with configurable fees
- **📉 Performance Metrics** - Sharpe ratio, max drawdown, cumulative returns

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wallstreetrl.git
cd wallstreetrl

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python untitled9.py
```

This will:
- Fetch historical data for AAPL (2019-present)
- Train a DQN agent for 25 episodes
- Save the trained model as `dqn_wallstreetrl.pt`

### Launch the Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` and start trading!

## 🎯 How It Works

### The RL Agent

The DQN agent learns to make trading decisions (Buy, Sell, Hold) by:
1. **Observing** market conditions through technical indicators
2. **Taking actions** to maximize portfolio value
3. **Learning** from rewards based on profit/loss
4. **Optimizing** its strategy through deep neural networks

### Architecture

```
State Space: Normalized features (RSI, MACD, Bollinger Bands, etc.)
Action Space: [Hold, Buy, Sell]
Reward: Change in portfolio value
Network: 3-layer MLP (128-128-64 neurons)
```

## 📊 Performance Comparison

The dashboard compares three strategies:

| Strategy | Description |
|----------|-------------|
| **🤖 RL Agent** | Deep Q-Network trained on historical data |
| **📊 Buy & Hold** | Simple baseline: buy at start, hold till end |
| **📉 SMA Strategy** | Trade on 20/50 moving average crossovers |

## 🛠️ Configuration

Customize your trading simulation in the dashboard sidebar:

- **Stock Ticker**: Any valid ticker (AAPL, MSFT, GOOGL, etc.)
- **Start Date**: Historical data range
- **Initial Balance**: Starting capital ($)
- **Transaction Cost**: Trading fees (0-1%)

## 📁 Project Structure

```
wallstreetrl/
├── app.py              # Streamlit dashboard
├── untitled9.py        # Training script & core logic
├── requirements.txt    # Python dependencies
└── dqn_wallstreetrl.pt # Trained model (generated)
```

## 🔧 Technical Details

### Features Used
- Price returns
- RSI (14-period)
- MACD with signal and histogram
- Bollinger Bands (width and position)
- Simple Moving Averages (20, 50)

### Hyperparameters
- Learning rate: 0.001
- Gamma (discount): 0.995
- Epsilon decay: 0.995 (exploration vs exploitation)
- Batch size: 128
- Replay buffer: 200,000 transitions

## 📈 Results

Typical performance on AAPL (2019-2024):
- **Sharpe Ratio**: 0.8-1.2
- **Max Drawdown**: 15-25%
- **Cumulative Return**: Competitive with benchmarks






---

**⭐ Star this repo if you find it useful!**
