import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from portfolio_env import DynamicPortfolioEnv
from stable_baselines3 import PPO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import json
import os
import warnings
import sqlite3
import time

warnings.filterwarnings('ignore')

# ============================================================
# DATABASE SETUP
# ============================================================

conn = sqlite3.connect('usage_log.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS simulation_log (
        timestamp TEXT,
        risk_level TEXT,
        tickers TEXT,
        duration REAL,
        final_value REAL,
        total_return REAL
    )
''')
conn.commit()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Portfolio Paper Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


components.html("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-G7B7RQM88L"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-G7B7RQM88L');
</script>
""", height=0)

st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 IntelliWealth AI-Powered Portfolio")
st.markdown("Real-time portfolio management using trained PPO reinforcement learning model")

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Portfolio Setup")
    default_tickers = "AAPL,MSFT,GOOGL,AMZN,META,JPM,BAC,UNH,JNJ,WMT"
    tickers_input = st.text_area(
        "Enter Tickers (comma-separated)",
        value=default_tickers,
        height=100
    )
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    st.subheader("Simulation Parameters")
    lookback_days = st.slider(
        "Lookback Period (days)",
        min_value=60,
        max_value=730,
        value=90
    )
    commission_rate = st.number_input(
        "Commission Rate (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01
    ) / 100
    
    st.subheader("Risk Preference")
    risk_level = st.radio(
        "Select Your Risk Preference",
        [
            "Low risk: Safe ETFs only",
            "Moderate risk: Stocks & ETFs",
            "High risk: All assets (incl. volatile)"
        ],
        key="risk_level"
    )
    
    run_button = st.button(
        "Start Paper Trading",
        type="primary",
        use_container_width=True
    )

# ============================================================
# MAIN EXECUTION
# ============================================================

if run_button:
    start_time = time.time()
    simulation_completed = False
    
    progress_container = st.empty()
    status_container = st.empty()
    
    # FIX: Store the original initial capital before any modifications
    original_initial_capital = initial_capital
    
    try:
        # Filter tickers based on risk level
        if risk_level == "Low risk: Safe ETFs only":
            safe_etfs = ["VOO", "SCHB", "IVV", "AGG", "BND", "VTI", "VTSAX"]
            ticker_list = [t for t in ticker_list if t in safe_etfs]
            if not ticker_list:
                st.warning("No safe ETFs found in your list. Using default safe ETFs.")
                ticker_list = ["VOO", "BND"]
        elif risk_level == "Moderate risk: Stocks & ETFs":
            avoid_high_risk = ["GME", "AMC", "MEME"]
            ticker_list = [t for t in ticker_list if t not in avoid_high_risk]
        
        st.info(f"Trading with {len(ticker_list)} assets: {', '.join(ticker_list)}")
        
        # Step 1: Load Model
        with progress_container.container():
            st.info("Loading AI model...")
        
        model_path = "notebooks/models/ppo_dynamic_portfolio_padded"
        if not os.path.exists(f"{model_path}.zip"):
            status_container.error(f"Model not found at {model_path}")
            st.stop()
        
        model = PPO.load(model_path)
        progress_container.success("✅ Model loaded successfully")
        
        # Load config
        config_path = os.path.join(os.path.dirname(model_path), "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                max_assets = config.get('max_assets', 100)
        else:
            max_assets = 100
        
        # Step 2: Download Data
        with status_container.container():
            st.info(f"Downloading {lookback_days} days of data for {len(ticker_list)} assets...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 100)
        
        data = yf.download(
            ticker_list,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
        
        # Handle single vs multiple tickers
        if 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        else:
            prices = data
        
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=ticker_list[0])
        
        prices = prices.ffill().dropna()
        
        if len(prices) < 60:
            progress_container.empty()
            status_container.error("Insufficient data. Try different tickers or longer lookback period.")
            st.stop()
        
        progress_container.success(f"✅ Downloaded {len(prices)} days of price data")
        
        # Step 3: Run Simulation
        with status_container.container():
            st.info("Running paper trading simulation...")
        
        env = DynamicPortfolioEnv(
            tickers=ticker_list,
            data=prices,
            initial_capital=original_initial_capital,
            commission=commission_rate,
            max_position_size=0.15,
            min_position_size=0.01,
            max_assets=max_assets
        )
        
        obs, _ = env.reset()
        
        # Initialize tracking
        portfolio_values = [original_initial_capital]
        dates_list = []
        weights_history = []
        cash_history = [original_initial_capital]
        holdings_value_history = [0]
        
        progress_bar = progress_container.progress(0)
        total_steps = len(prices) - 61
        
        # Run simulation
        for i in range(60, len(prices) - 1):
            # Get observation
            obs = env.feature_engineer.get_features(prices, i)
            
            # Model prediction
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, _, info = env.step(action)
            
            # Track results
            portfolio_values.append(float(info['portfolio_value']))
            dates_list.append(prices.index[i + 1])
            weights_history.append(action[:len(ticker_list)])
            cash_history.append(float(env.cash))
            holdings_value = env.holdings @ prices.iloc[i + 1].values
            holdings_value_history.append(float(holdings_value))
            
            # Update progress
            progress = (i - 59) / total_steps
            progress_bar.progress(progress)
            status_container.info(f"Processing {prices.index[i].strftime('%Y-%m-%d')} ({i-59}/{total_steps})")
            
            if done:
                break
        
        progress_container.empty()
        status_container.success("✅ Simulation complete!")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': dates_list,
            'portfolio_value': portfolio_values[1:],
            'cash': cash_history[1:],
            'holdings_value': holdings_value_history[1:]
        }).set_index('date')
        
        weights_df = pd.DataFrame(
            weights_history,
            columns=ticker_list,
            index=dates_list
        )
        
    except Exception as e:
        progress_container.empty()
        status_container.error(f"Error during simulation: {str(e)}")
        st.exception(e)
        st.stop()
    
    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    
    st.markdown("---")
    
    # Calculate metrics - FIX: Use original_initial_capital consistently
    daily_returns = results_df['portfolio_value'].pct_change().dropna()
    total_return = (results_df['portfolio_value'].iloc[-1] / original_initial_capital - 1) * 100
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    max_drawdown = ((results_df['portfolio_value'].cummax() - results_df['portfolio_value']) / results_df['portfolio_value'].cummax()).max() * 100
    volatility = daily_returns.std() * np.sqrt(252) * 100
    winning_days = (daily_returns > 0).sum()
    total_days = len(daily_returns)
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    simulation_completed = True
    
    # Log to database
    c.execute(
        'INSERT INTO simulation_log VALUES (?, ?, ?, ?, ?, ?)',
        (
            time.strftime('%Y-%m-%d %H:%M:%S'),
            risk_level,
            ','.join(ticker_list),
            duration,
            results_df['portfolio_value'].iloc[-1],
            total_return
        )
    )
    conn.commit()
    
    # Display metrics
    st.markdown("""
        <style>
            .metric-container {
                background-color: #0e1117;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #262730;
                text-align: center;
            }
            .metric-label {
                font-size: 14px;
                color: #a3a8b4;
                margin-bottom: 8px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .metric-value {
                font-size: 28px;
                font-weight: 700;
                color: #fafafa;
                margin-bottom: 4px;
            }
            .metric-delta {
                font-size: 16px;
                font-weight: 600;
            }
            .delta-positive { color: #21c55d; }
            .delta-negative { color: #ef4444; }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_color = "delta-positive" if total_return > 0 else "delta-negative"
        delta_symbol = "+" if total_return > 0 else ""
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Final Value</div>
                <div class="metric-value">${results_df['portfolio_value'].iloc[-1]:,.0f}</div>
                <div class="metric-delta {delta_color}">{delta_symbol}{abs(total_return):.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{total_return:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value" style="color: {'#21c55d' if sharpe_ratio > 0 else '#ef4444'}">{sharpe_ratio:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" style="color: #ef4444">-{max_drawdown:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        win_rate = (winning_days / total_days) * 100
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{win_rate:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 Allocation", "📋 Statistics", "📥 Export"])
    
    with tab1:
        st.subheader("Portfolio Performance")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Portfolio Value Over Time", "Drawdown"),
            vertical_spacing=0.12,
            row_heights=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#2E86AB', width=2)
            ),
            row=1, col=1
        )
        fig.add_hline(
            y=original_initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital",
            row=1, col=1
        )
        
        drawdown = (results_df['portfolio_value'].cummax() - results_df['portfolio_value']) / results_df['portfolio_value'].cummax() * 100
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=-drawdown,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_layout(
            height=600,
            showlegend=False,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Portfolio Allocation Over Time")
        
        fig = go.Figure()
        for ticker in ticker_list:
            fig.add_trace(
                go.Scatter(
                    x=weights_df.index,
                    y=weights_df[ticker],
                    mode='lines',
                    name=ticker,
                    stackgroup='one',
                    fillcolor=None
                )
            )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Portfolio Weight",
            hovermode='x unified',
            height=500,
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Average Portfolio Weights")
        avg_weights = weights_df.mean().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=avg_weights.values * 100,
                        y=avg_weights.index,
                        orientation='h',
                        marker_color='#2E86AB'
                    )
                ]
            )
            fig.update_layout(
                xaxis_title="Weight (%)",
                yaxis_title="Ticker",
                height=400,
                showlegend=False,
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                pd.DataFrame({
                    'Ticker': avg_weights.index,
                    'Avg Weight (%)': [f"{w*100:.2f}" for w in avg_weights.values],
                    'Min Weight (%)': [f"{weights_df[t].min()*100:.2f}" for t in avg_weights.index],
                    'Max Weight (%)': [f"{weights_df[t].max()*100:.2f}" for t in avg_weights.index]
                }).set_index('Ticker'),
                height=400
            )
    
    with tab3:
        st.subheader("Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Portfolio Metrics")
            # FIX: Use original_initial_capital in metrics table
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Initial Capital',
                    'Final Value',
                    'Total Return (%)',
                    'Annualized Return (%)',
                    'Sharpe Ratio',
                    'Max Drawdown (%)',
                    'Volatility Annual (%)',
                    'Best Day (%)',
                    'Worst Day (%)',
                    'Avg Daily Return (bps)',
                    'Win Rate (%)',
                    'Total Trading Days'
                ],
                'Value': [
                    f"{original_initial_capital:,.2f}",
                    f"{results_df['portfolio_value'].iloc[-1]:,.2f}",
                    f"{total_return:.2f}",
                    f"{total_return * 252 / lookback_days:.2f}",
                    f"{sharpe_ratio:.2f}",
                    f"{max_drawdown:.2f}",
                    f"{volatility:.2f}",
                    f"{daily_returns.max()*100:.2f}",
                    f"{daily_returns.min()*100:.2f}",
                    f"{daily_returns.mean()*10000:.4f}",
                    f"{winning_days/total_days*100:.2f}",
                    f"{total_days}"
                ]
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("### Daily Returns Distribution")
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=daily_returns * 100,
                    nbinsx=50,
                    marker_color='#2E86AB',
                    name='Daily Returns'
                )
            )
            fig.add_vline(
                x=daily_returns.mean() * 100,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {daily_returns.mean()*100:.3f}%"
            )
            fig.update_layout(
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                height=400,
                showlegend=False,
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Download Portfolio Data")
            csv = results_df.to_csv()
            st.download_button(
                label="📥 Download Portfolio Values CSV",
                data=csv,
                file_name=f"portfolio_paper_trading_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.markdown("### Download Allocation Data")
            weights_csv = weights_df.to_csv()
            st.download_button(
                label="📥 Download Weights History CSV",
                data=weights_csv,
                file_name=f"weights_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

else:
    st.info("Configure your portfolio parameters in the sidebar and click **Start Paper Trading** to begin!")
    
    st.subheader("Dashboard Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Portfolio Performance")
        st.markdown("""
        - Real-time portfolio value tracking
        - Cumulative return visualization
        - Drawdown analysis
        - Performance metrics (Sharpe, volatility, max drawdown)
        """)
    
    with col2:
        st.markdown("### Asset Allocation")
        st.markdown("""
        - Dynamic weight adjustments over time
        - Stacked area chart visualization
        - Average, min, max weight statistics
        - Export allocation history
        """)
