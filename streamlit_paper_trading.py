"""
Streamlit Paper Trading Dashboard for Adaptive Portfolio Manager
Run with: streamlit run streamlit_paper_trading.py
Updated for Python 3.13 + Streamlit Cloud compatibility
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from portfolio_env import DynamicPortfolioEnv
from stable_baselines3 import PPO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import warnings
import sqlite3
import time

warnings.filterwarnings('ignore')

# ============= DATABASE SETUP =============
try:
    conn = sqlite3.connect("usage_log.db", check_same_thread=False)
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
except Exception as e:
    st.warning(f"Database warning (non-critical): {e}")

# ============= PAGE CONFIGURATION =============
st.set_page_config(
    page_title="IntelliWealth: AI-Powered Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CUSTOM CSS =============
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
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
    .delta-positive {
        color: #21c55d;
    }
    .delta-negative {
        color: #ef4444;
    }
    </style>
""", unsafe_allow_html=True)

# ============= TITLE =============
st.title("🤖 IntelliWealth: AI-Powered Portfolio")
st.markdown("**Real-time portfolio management using trained PPO reinforcement learning model**")

# ============= SIDEBAR CONFIGURATION =============
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.form("config_form"):
        # Portfolio Setup
        st.subheader("Portfolio Setup")
        default_tickers = "AAPL,MSFT,GOOGL,AMZN,META,JPM,BAC,UNH,JNJ,WMT"
        tickers_input = st.text_area(
            "Enter Tickers (comma-separated)", 
            value=default_tickers, 
            height=100,
            help="Enter stock tickers separated by commas"
        )
        ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        st.info(f"📊 Trading with {len(ticker_list)} assets: {', '.join(ticker_list)}")
        
        initial_capital = st.number_input(
            "Initial Capital ($)", 
            min_value=10000, 
            max_value=10000000, 
            value=1000000,
            step=100000
        )
        
        # Simulation Parameters
        st.subheader("Simulation Parameters")
        lookback_days = st.slider(
            "Lookback Period (days)", 
            min_value=60, 
            max_value=730, 
            value=251,
            step=10,
            help="Number of historical days to analyze"
        )
        
        commission_rate = st.number_input(
            "Commission Rate (%)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.10,
            step=0.01,
            help="Trading commission percentage"
        ) / 100
        
        # Risk Preference
        st.subheader("Risk Preference")
        risk_level = st.radio(
            "Select Your Risk Preference",
            [
                "Low risk (Safe ETFs only)", 
                "Moderate risk (Stocks & ETFs)", 
                "High risk (All assets incl. volatile)"
            ],
            key="risk_level"
        )
        
        run_button = st.form_submit_button("🚀 Start Paper Trading", use_container_width=True)

# ============= MAIN CONTENT =============
if run_button:
    start_time = time.time()
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        # STEP 1: VALIDATE TICKERS
        if not ticker_list or len(ticker_list) == 0:
            status_container.error("❌ Please enter at least one ticker symbol")
            st.stop()
        
        # Apply risk filters
        if risk_level == "Low risk (Safe ETFs only)":
            safe_etfs = ["VOO", "SCHB", "IVV", "AGG", "BND", "VTI", "VTSAX", "SPLG", "SPYG"]
            ticker_list = [t for t in ticker_list if t in safe_etfs]
            if not ticker_list:
                status_container.warning("⚠️ No safe ETFs found. Using default safe ETFs.")
                ticker_list = ["VOO", "BND"]
                
        elif risk_level == "Moderate risk (Stocks & ETFs)":
            avoid_high_risk = ["GME", "AMC", "MEME", "NVDA"]
            ticker_list = [t for t in ticker_list if t not in avoid_high_risk]
        
        st.info(f"✅ Trading with {len(ticker_list)} assets: {', '.join(ticker_list)}")
        
        # STEP 2: DOWNLOAD MARKET DATA
        progress_container.info("📊 Downloading historical market data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 100)
        
        try:
            data = yf.download(
                ticker_list, 
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d'),
                auto_adjust=True, 
                progress=False,
                threads=True,
                timeout=30
            )
            
            # Handle single vs multiple tickers
            if len(ticker_list) == 1:
                data = data.to_frame()
                data.columns = pd.MultiIndex.from_product([data.columns, ticker_list])
            
            # Extract closing prices
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"]
            else:
                prices = data
            
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=ticker_list[0])
            
            # Handle missing values (Python 3.13 compatible)
            prices = prices.ffill().bfill().dropna()
            
            # Validate data
            if len(prices) < 60:
                status_container.error(
                    f"❌ **Insufficient data**: Only {len(prices)} days available.\n"
                    f"Try:\n"
                    f"• Different tickers (AAPL, MSFT, GOOGL work reliably)\n"
                    f"• Longer lookback period (500+ days)\n"
                    f"• Check tickers are valid on Yahoo Finance"
                )
                st.stop()
            
            progress_container.success(f"✅ Downloaded {len(prices)} trading days")
            
        except Exception as download_error:
            status_container.error(
                f"❌ **Data download failed**: {str(download_error)}\n\n"
                f"Try:\n"
                f"• Check ticker symbols (e.g., AAPL, MSFT)\n"
                f"• Use different tickers\n"
                f"• Reduce lookback period"
            )
            st.stop()
        
        # STEP 3: LOAD MODEL
        progress_container.info("🤖 Loading AI model...")
        
        model_paths = [
            "notebooks/models/ppo_dynamic_portfolio_padded",
            "models/ppo_dynamic_portfolio_padded",
            "ppo_dynamic_portfolio_padded"
        ]
        
        model = None
        model_path = None
        
        for path in model_paths:
            if os.path.exists(path + ".zip"):
                try:
                    model = PPO.load(path)
                    model_path = path
                    break
                except Exception:
                    continue
        
        if model is None:
            status_container.warning(
                "⚠️ **Model not found** - Using fallback strategy\n\n"
                "Using equal-weight portfolio for this session."
            )
            use_fallback = True
        else:
            status_container.success("✅ Model loaded successfully")
            use_fallback = False
            
            config_path = os.path.join(os.path.dirname(model_path), "training_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                max_assets = config.get("max_assets", len(ticker_list))
            else:
                max_assets = len(ticker_list)
        
        # STEP 4: CREATE ENVIRONMENT
        progress_container.info("⚙️ Initializing portfolio environment...")
        
        try:
            env = DynamicPortfolioEnv(
                tickers=ticker_list,
                data=prices,
                initial_capital=initial_capital,
                commission=commission_rate,
                max_position_size=0.20,
                min_position_size=0.01,
                max_assets=len(ticker_list)
            )
            
            obs, _ = env.reset()
            progress_container.success("✅ Environment initialized")
            
        except Exception as env_error:
            status_container.error(f"❌ Environment creation failed: {str(env_error)}")
            st.stop()
        
        # STEP 5: RUN SIMULATION
        progress_container.info("🔄 Running portfolio simulation...")
        
        portfolio_values = [initial_capital]
        dates = [prices.index[60]]
        weights_history = []
        cash_history = [initial_capital]
        
        progress_bar = progress_container.progress(0)
        total_steps = len(prices) - 61
        
        for i in range(60, len(prices) - 1):
            try:
                obs = env.feature_engineer.get_features(prices, i)
                
                if use_fallback:
                    action = np.ones(len(ticker_list)) / len(ticker_list)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, done, _, info = env.step(action)
                
                portfolio_values.append(info['portfolio_value'])
                dates.append(prices.index[i + 1])
                weights_history.append(action[:len(ticker_list)])
                cash_history.append(env.cash)
                
                progress = (i - 59) / total_steps
                progress_bar.progress(min(progress, 0.99))
                
                if i % 30 == 0:
                    status_container.info(
                        f"🔄 Processing: {prices.index[i].strftime('%Y-%m-%d')} "
                        f"({i-59}/{total_steps}) | Portfolio: ${portfolio_values[-1]:,.0f}"
                    )
                
                if done:
                    break
                    
            except Exception as step_error:
                st.warning(f"Step {i} warning (continuing): {str(step_error)}")
                continue
        
        progress_bar.progress(1.0)
        progress_container.success("✅ Simulation complete!")
        
        # STEP 6: PROCESS RESULTS
        results_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values,
            'cash': cash_history
        }).set_index('date')
        
        weights_df = pd.DataFrame(
            weights_history, 
            columns=ticker_list, 
            index=dates[:-1]
        )
        
        # Calculate metrics
        daily_returns = results_df['portfolio_value'].pct_change().dropna()
        total_return = (results_df['portfolio_value'].iloc[-1] / initial_capital - 1) * 100
        
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        max_drawdown = ((results_df['portfolio_value'].cummax() - results_df['portfolio_value']) / 
                        results_df['portfolio_value'].cummax()).max() * 100
        volatility = daily_returns.std() * np.sqrt(252) * 100
        winning_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = (winning_days / total_days) * 100 if total_days > 0 else 0
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        # Log to database
        try:
            c.execute("INSERT INTO simulation_log VALUES (?, ?, ?, ?, ?, ?)", 
                (
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    risk_level,
                    ','.join(ticker_list),
                    duration,
                    results_df['portfolio_value'].iloc[-1],
                    total_return
                )
            )
            conn.commit()
        except:
            pass
        
    except Exception as e:
        progress_container.empty()
        status_container.error(f"❌ Unexpected error: {str(e)}")
        st.exception(e)
        st.stop()
    
    # ============= DISPLAY RESULTS =============
    st.markdown("---")
    st.success(f"✅ Simulation completed in {duration}s")
    
    # Metrics display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_color = "delta-positive" if total_return > 0 else "delta-negative"
        delta_symbol = "↑" if total_return > 0 else "↓"
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Final Value</div>
                <div class="metric-value">${results_df['portfolio_value'].iloc[-1]:,.0f}</div>
                <div class="metric-delta {delta_color}">{delta_symbol} {abs(total_return):.2f}%</div>
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
                <div class="metric-value" style="color: {'#21c55d' if sharpe_ratio > 0 else '#ef4444'};">{sharpe_ratio:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" style="color: #ef4444;">-{max_drawdown:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{win_rate:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "💼 Allocation", "📊 Statistics", "📥 Export"])
    
    with tab1:
        st.subheader("Portfolio Performance")
        
        fig = make_subplots(
            rows=2, cols=1, 
            subplot_titles=('Portfolio Value Over Time', 'Drawdown'),
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
            y=initial_capital, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Initial Capital", 
            row=1, col=1
        )
        
        drawdown = ((results_df['portfolio_value'].cummax() - results_df['portfolio_value']) / 
                   results_df['portfolio_value'].cummax()) * 100
        
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
            fig.add_trace(go.Scatter(
                x=weights_df.index,
                y=weights_df[ticker],
                mode='lines',
                name=ticker,
                stackgroup='one'
            ))
        
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
            fig = go.Figure(data=[
                go.Bar(
                    x=avg_weights.values * 100, 
                    y=avg_weights.index,
                    orientation='h',
                    marker_color='#2E86AB'
                )
            ])
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
                    'Avg Weight': [f"{w*100:.2f}%" for w in avg_weights.values],
                    'Min Weight': [f"{weights_df[t].min()*100:.2f}%" for t in avg_weights.index],
                    'Max Weight': [f"{weights_df[t].max()*100:.2f}%" for t in avg_weights.index]
                }).set_index('Ticker'),
                height=400,
                use_container_width=True
            )
    
    with tab3:
        st.subheader("Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Portfolio Metrics")
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Initial Capital', 'Final Value', 'Total Return', 'Annualized Return',
                    'Sharpe Ratio', 'Max Drawdown', 'Volatility (Annual)',
                    'Best Day', 'Worst Day', 'Avg Daily Return', 'Win Rate', 'Total Trading Days'
                ],
                'Value': [
                    f"${initial_capital:,.2f}",
                    f"${results_df['portfolio_value'].iloc[-1]:,.2f}",
                    f"{total_return:.2f}%",
                    f"{(total_return / (total_days / 252)):.2f}%",
                    f"{sharpe_ratio:.2f}",
                    f"{max_drawdown:.2f}%",
                    f"{volatility:.2f}%",
                    f"{daily_returns.max()*100:.2f}%",
                    f"{daily_returns.min()*100:.2f}%",
                    f"{daily_returns.mean()*100:.4f}%",
                    f"{win_rate:.2f}%",
                    f"{total_days}"
                ]
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("### Daily Returns Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=daily_returns * 100,
                nbinsx=50,
                marker_color='#2E86AB',
                name='Daily Returns'
            ))
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
                label="📥 Download Portfolio Values (CSV)",
                data=csv,
                file_name=f"portfolio_paper_trading_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("### Download Allocation Data")
            weights_csv = weights_df.to_csv()
            st.download_button(
                label="📥 Download Weights History (CSV)",
                data=weights_csv,
                file_name=f"weights_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("👈 Configure your portfolio parameters in the sidebar and click '🚀 Start Paper Trading' to begin")
    
    st.subheader("📊 Dashboard Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Portfolio Performance")
        st.markdown("""
        - Real-time portfolio value tracking
        - Cumulative return visualization
        - Drawdown analysis
        - Performance metrics (Sharpe, volatility, max drawdown)
        """)
    
    with col2:
        st.markdown("### 💼 Asset Allocation")
        st.markdown("""
        - Dynamic weight adjustments over time
        - Stacked area chart visualization
        - Average, min, max weight statistics
        - Export allocation history
        """)
    
    with st.expander("ℹ️ How to Use", expanded=True):
        st.markdown("""
        ### Getting Started
        
        1. **Configure Portfolio**: Enter your desired tickers (e.g., AAPL,MSFT,GOOGL)
        2. **Set Parameters**: Choose initial capital, lookback period, and commission rate
        3. **Select Risk Level**: Choose your risk preference (Low/Moderate/High)
        4. **Run Simulation**: Click the '🚀 Start Paper Trading' button
        5. **Analyze Results**: View performance metrics, allocation charts, and statistics
        6. **Export Data**: Download results as CSV for further analysis
        
        ### Features
        
        - **AI-Powered Decisions**: Uses trained PPO reinforcement learning model
        - **Risk Appetite Control**: Filter assets based on your risk preference
        - **Real Market Data**: Downloads historical prices from Yahoo Finance
        - **Professional Metrics**: Sharpe ratio, max drawdown, volatility, win rate
        - **Interactive Charts**: Plotly visualizations for performance and allocation
        - **Export Ready**: Download CSV files for Excel/Python analysis
        
        ### Risk Levels
        
        - **Low Risk**: Only safe ETFs (VOO, BND, VTI, etc.)
        - **Moderate Risk**: Broad stocks & ETFs (excludes meme stocks)
        - **High Risk**: All assets including volatile stocks
        
        ### Troubleshooting
        
        **"Insufficient data" error?**
        - Use different tickers (AAPL, MSFT, GOOGL are reliable)
        - Increase lookback period (500+ days)
        - Verify tickers exist on Yahoo Finance
        
        **"Model not found" warning?**
        - This is normal on first deployment
        - System will use equal-weight portfolio
        - Model loads on subsequent refreshes
        """)
