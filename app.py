import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Import functions from untitled9
from untitled9 import (
    prepare_dataset,
    simulate_agent,
    buy_and_hold,
    sma_strategy,
    compute_metrics
)

st.set_page_config(page_title="WallStreetRL", layout="wide")

st.title("📈 WallStreetRL — RL Trading Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
initial_balance = st.sidebar.number_input("Initial Balance", value=10000, step=1000)
transaction_cost = st.sidebar.slider("Transaction Cost", 0.0, 0.01, 0.001, format="%.4f")

run_btn = st.sidebar.button("Run Simulation")

# ---------------- Main Logic ----------------
if run_btn:
    with st.spinner("Fetching data & running agent..."):
        try:
            # Prepare dataset
            train_df, test_df = prepare_dataset(
                ticker=ticker,
                start=str(start_date)
            )
            
            # Get feature columns (now ending with _norm_)
            feature_cols = [c for c in test_df.columns if c.endswith("_norm_")]
            
            # Check if model file exists
            import os
            if not os.path.exists("dqn_wallstreetrl.pt"):
                st.error("Model file 'dqn_wallstreetrl.pt' not found. Please train the model first by running untitled9.py")
                st.stop()
            
            # Run RL agent simulation
            rl_res = simulate_agent(
                test_df,
                feature_cols,
                agent_path="dqn_wallstreetrl.pt",
                initial_balance=initial_balance,
                transaction_cost=transaction_cost
            )
            
            # Run baseline strategies
            bah_res = buy_and_hold(test_df, initial_balance, transaction_cost)
            sma_res = sma_strategy(test_df, initial_balance=initial_balance, transaction_cost=transaction_cost)
            
        except Exception as e:
            st.error(f"Error during simulation: {str(e)}")
            st.stop()
    
    # ---------------- Metrics ----------------
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🤖 RL Agent")
        st.metric("Final Value", f"${rl_res['value'].iloc[-1]:,.2f}")
        metrics_rl = compute_metrics(rl_res['value'])
        for k, v in metrics_rl.items():
            if k == 'Total Profit':
                st.write(f"**{k}**: ${v:,.2f}")
            elif k == 'Cumulative Return' or k == 'Max Drawdown':
                st.write(f"**{k}**: {v:.2%}")
            else:
                st.write(f"**{k}**: {v:.4f}")
    
    with col2:
        st.markdown("### 📊 Buy & Hold")
        st.metric("Final Value", f"${bah_res['value'].iloc[-1]:,.2f}")
        metrics_bah = compute_metrics(bah_res['value'])
        for k, v in metrics_bah.items():
            if k == 'Total Profit':
                st.write(f"**{k}**: ${v:,.2f}")
            elif k == 'Cumulative Return' or k == 'Max Drawdown':
                st.write(f"**{k}**: {v:.2%}")
            else:
                st.write(f"**{k}**: {v:.4f}")
    
    with col3:
        st.markdown("### 📉 SMA Strategy")
        st.metric("Final Value", f"${sma_res['value'].iloc[-1]:,.2f}")
        metrics_sma = compute_metrics(sma_res['value'])
        for k, v in metrics_sma.items():
            if k == 'Total Profit':
                st.write(f"**{k}**: ${v:,.2f}")
            elif k == 'Cumulative Return' or k == 'Max Drawdown':
                st.write(f"**{k}**: {v:.2%}")
            else:
                st.write(f"**{k}**: {v:.4f}")
    
    # ---------------- Plots ----------------
    st.subheader("Portfolio Value Comparison")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rl_res.index, rl_res["value"], label="RL Agent", linewidth=2)
    ax.plot(bah_res.index, bah_res["value"], label="Buy & Hold", linestyle="--", linewidth=2)
    ax.plot(sma_res.index, sma_res["value"], label="SMA Strategy", linestyle=":", linewidth=2)
    ax.legend(fontsize=10)
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # ---------------- RL Trades ----------------
    st.subheader("RL Agent Buy/Sell Actions")
    
    buys = rl_res[rl_res["action"] == 1]
    sells = rl_res[rl_res["action"] == 2]
    
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(rl_res.index, rl_res["price"], color="black", label="Price", linewidth=1.5)
    ax2.scatter(buys.index, buys["price"], marker="^", color="green", s=100, label="Buy", zorder=5)
    ax2.scatter(sells.index, sells["price"], marker="v", color="red", s=100, label="Sell", zorder=5)
    ax2.legend(fontsize=10)
    ax2.set_ylabel("Price ($)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    
    # ---------------- Additional Stats ----------------
    st.subheader("Trading Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_buys = len(buys)
        num_sells = len(sells)
        st.metric("RL Total Trades", num_buys + num_sells)
        st.write(f"Buys: {num_buys}, Sells: {num_sells}")
    
    with col2:
        bah_trades = len(bah_res[bah_res["action"] != 0])
        st.metric("Buy & Hold Trades", bah_trades)
    
    with col3:
        sma_trades = len(sma_res[sma_res["action"] != 0])
        st.metric("SMA Trades", sma_trades)

else:
    st.info("👈 Set parameters in the sidebar and click **Run Simulation** to begin")
    
    st.markdown("""
    ### About WallStreetRL
    
    This dashboard compares a Deep Q-Network (DQN) reinforcement learning agent against traditional trading strategies:
    
    - **🤖 RL Agent**: Uses deep reinforcement learning to learn optimal trading decisions
    - **📊 Buy & Hold**: Simple baseline that buys and holds the stock
    - **📉 SMA Strategy**: Trades based on 20/50 simple moving average crossovers
    
    #### How to Use:
    1. Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)
    2. Set the start date for historical data
    3. Configure initial balance and transaction costs
    4. Click "Run Simulation" to see results
    
    ⚠️ **Note**: Make sure you have trained the model first by running `untitled9.py`
    """)