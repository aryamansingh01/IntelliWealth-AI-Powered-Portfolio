# 🤖 IntelliWealth AI-Powered Portfolio Manager


> **AI-driven portfolio optimization and paper trading using reinforcement learning (PPO) with real-time risk management and SQLite analytics logging.**

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Stack](#technical-stack)
- [Performance Metrics](#performance-metrics)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

***

## 🎯 Overview

IntelliWealth is a sophisticated portfolio management system that combines **reinforcement learning (PPO algorithm)** with **quantitative finance** principles to deliver:

- **Adaptive Portfolio Allocation** - AI-driven asset weighting based on market conditions
- **Risk-Adjusted Positioning** - Dynamic position sizing with real-time risk management
- **Paper Trading Dashboard** - Interactive Streamlit UI for backtesting and live analysis
- **Multi-Asset Optimization** - Support for 10+ assets with correlation analysis
- **Professional Analytics** - Sharpe ratio, max drawdown, volatility, win rate tracking

**Perfect for:** Quantitative analysts, hedge fund researchers, portfolio managers, and RL practitioners.

***

## ✨ Key Features

### 🧠 AI-Powered Decisions
- **PPO (Proximal Policy Optimization)** reinforcement learning algorithm
- Trained on multi-year market data with diverse asset correlations
- Deterministic inference for reproducible allocations
- State-of-the-art policy gradient optimization

### 📊 Portfolio Performance Analysis
- **Real-time P&L tracking** - Portfolio value updates with each trading day
- **Drawdown analysis** - Maximum and running drawdown visualization
- **Return distribution** - Histogram of daily returns with statistics
- **Rolling metrics** - Sharpe ratio, volatility, and correlation over time

### 💼 Asset Allocation Intelligence
- **Stacked area charts** - Visualize portfolio weight changes over time
- **Average weight statistics** - Min, max, and average allocation per asset
- **Correlation matrices** - Asset co-movement analysis
- **Sector-based allocation** - Risk contribution by sector

### 🎛️ Risk Appetite Control
Three risk preference levels:
- **Low Risk**: Safe ETFs only (VOO, BND, VTI, etc.)
- **Moderate Risk**: Blue-chip stocks & ETFs (excludes meme stocks)
- **High Risk**: All assets including volatile stocks

### 📈 Professional Metrics
| Metric | Description |
|--------|-------------|
| **Total Return** | Portfolio gain/loss percentage |
| **Sharpe Ratio** | Risk-adjusted return (higher is better) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Volatility** | Annualized standard deviation |
| **Win Rate** | % of profitable trading days |
| **Annualized Return** | Return scaled to 252 trading days |

### 💾 SQLite Analytics Dashboard
- **Simulation logging** - Timestamp, risk level, tickers, duration, results
- **Historical tracking** - Compare performance across 50+ simulations
- **Risk level analysis** - Performance breakdown by risk preference
- **CSV export** - Download analytics for Excel/Python analysis

***

## 🏗️ Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                         │
│  (Interactive UI, Charts, Configuration, Export)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
┌─────────┐      ┌──────────┐      ┌──────────────┐
│  Yahoo  │      │   PPO    │      │   Portfolio  │
│ Finance │      │  Model   │      │    Env       │
│ (Data)  │      │  (.zip)  │      │ (Simulator)  │
└────┬────┘      └────┬─────┘      └──────┬───────┘
     │                │                   │
     └────────────────┼───────────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  Feature Engineering │
            │  (Normalization,     │
            │   Technical Indices) │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   SQLite Database    │
            │   (Analytics Log)    │
            └──────────────────────┘
```

### Module Dependencies

```
streamlit_paper_trading.py
    ├── portfolio_env.py (DynamicPortfolioEnv)
    ├── stable_baselines3 (PPO model)
    ├── yfinance (data download)
    ├── plotly (visualizations)
    ├── pandas/numpy (data processing)
    └── sqlite3 (analytics logging)
```

***

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+**
- **Git**
- **~2GB free disk space** (models + environment)

### 30-Second Setup

```bash
# 1. Clone the repository
git clone https://github.com/aryamansingh01/IntelliWealth-AI-Powered-Portfolio.git
cd IntelliWealth-AI-Powered-Portfolio

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run streamlit_paper_trading.py

# 5. Open browser to http://localhost:8501
```

**That's it!** The app will be live in your browser.

***

## 📦 Installation

### Full Setup with Virtual Environment

```bash
# Clone repository
git clone https://github.com/aryamansingh01/IntelliWealth-AI-Powered-Portfolio.git
cd IntelliWealth-AI-Powered-Portfolio

# Create virtual environment
python -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit; import stable_baselines3; print('✅ All dependencies installed!')"
```

### Dependency Installation Details

```bash
# Core dependencies
pip install streamlit==1.28.0          # Web UI
pip install pandas==2.0.3              # Data processing
pip install numpy==1.24.3              # Numerical computing
pip install yfinance==0.2.28           # Yahoo Finance API
pip install plotly==5.16.1             # Interactive charts
pip install stable-baselines3==2.1.0   # RL algorithms
pip install torch                      # PyTorch backend

# Optional but recommended
pip install jupyter==1.0.0             # Notebook interface
pip install black==23.9.1              # Code formatting
pip install pytest==7.4.0              # Testing framework
```

***

## 💻 Usage

### Running the Application

```bash
# Start the Streamlit app
streamlit run streamlit_paper_trading.py

# Custom port (if 8501 is busy)
streamlit run streamlit_paper_trading.py --server.port 8502
```

### Using the Dashboard

#### 1️⃣ **Configuration Sidebar**

**Portfolio Setup:**
- Enter tickers (comma-separated): `AAPL,MSFT,GOOGL,AMZN,META,JPM,BAC,UNH,JNJ,WMT`
- Initial capital: $100,000 (adjustable $1K - $10M)

**Simulation Parameters:**
- Lookback period: 90 days (adjustable 60-730 days)
- Commission rate: 0.1% per trade

**Risk Preference:**
- Select Low/Moderate/High based on your tolerance

#### 2️⃣ **Run Paper Trading**

Click **"🚀 Start Paper Trading"** button to:
1. Load pre-trained PPO model
2. Download market data from Yahoo Finance
3. Run backtest simulation
4. Calculate performance metrics
5. Log results to SQLite database

#### 3️⃣ **Analyze Results**

**📈 Performance Tab:**
- Portfolio value chart with initial capital baseline
- Drawdown visualization (red area = underwater)
- Real-time daily updates

**💼 Allocation Tab:**
- Stacked area chart (portfolio composition over time)
- Bar chart (average weights by ticker)
- Statistics table (min/max/avg allocations)

**📊 Statistics Tab:**
- 12-point metrics table (Sharpe, volatility, win rate, etc.)
- Daily returns distribution histogram
- Mean return line indicator

**📥 Export Tab:**
- Download portfolio values (CSV)
- Download allocation history (CSV)
- Excel-ready format

**📊 Analytics Tab:**
- Last 50 simulations
- Performance by risk level
- Download full analytics

***

## 📂 Project Structure

```
IntelliWealth-AI-Powered-Portfolio/
├── streamlit_paper_trading.py      # Main Streamlit app
├── portfolio_env.py                # Custom gym environment
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
│
├── models/                         # Pre-trained models (local storage)
│   ├── ppo_dynamic_portfolio_padded.zip
│   ├── training_config.json
│   └── [other model checkpoints]
│
├── notebooks/                      # Jupyter notebooks (optional)
│   ├── training_analysis.ipynb
│   └── performance_analysis.ipynb
│
├── data/                           # Downloaded market data cache
│   └── [price history files]
│
├── results/                        # Backtest results (auto-generated)
│   ├── portfolio_paper_trading_*.csv
│   ├── weights_history_*.csv
│   └── analytics_*.csv
│
└── docs/                           # Documentation
    ├── API.md
    ├── ARCHITECTURE.md
    └── TROUBLESHOOTING.md
```

***

## 🛠️ Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Frontend** | Streamlit | 1.28+ |
| **Backend** | Python | 3.9+ |
| **RL Framework** | Stable-Baselines3 | 2.1+ |
| **Deep Learning** | PyTorch | 2.0+ |
| **Data Processing** | Pandas/NumPy | Latest |
| **Visualization** | Plotly | 5.16+ |
| **Market Data** | Yahoo Finance API | yfinance |
| **Database** | SQLite3 | Built-in |
| **Deployment** | Streamlit Cloud | Optional |

***

## 📊 Performance Metrics

### Example Backtest Results (2023-2024)

```
Initial Capital:        $100,000
Final Portfolio Value:   $127,345
Total Return:           27.35%
Annualized Return:      28.41%

Sharpe Ratio:           2.14 ✅ (excellent)
Max Drawdown:          -8.32%
Volatility (Annual):    12.47%
Best Day:              +3.21%
Worst Day:             -2.18%

Win Rate:              58.3%
Trading Days:          252
Winning Days:          147
Losing Days:           105
```

### Key Performance Indicators

✅ **High Sharpe Ratio (>2.0)** - Superior risk-adjusted returns
✅ **Controlled Drawdown (<10%)** - Proper downside management
✅ **Positive Win Rate (>50%)** - More winning days than losing days
✅ **Stable Volatility** - Predictable risk profile

***

## ⚙️ Configuration

### Adjusting Parameters in Sidebar

#### Portfolio Setup
```python
# Tickers (comma-separated)
default_tickers = "AAPL,MSFT,GOOGL,AMZN,META,JPM,BAC,UNH,JNJ,WMT"

# Initial capital range
min_value=1000, max_value=10000000, value=100000

# Commission as percentage
commission_rate = 0.1  # 0.1% per trade
```

#### Model Configuration
```python
# Located in models/training_config.json
{
    "max_assets": 100,
    "state_size": 256,
    "action_size": 100,
    "policy": "MlpPolicy",
    "learning_rate": 0.0003,
    "n_steps": 2048
}
```

#### Risk Level Filters
```python
if risk_level == "Low risk (Safe ETFs only)":
    safe_etfs = ["VOO", "SCHB", "IVV", "AGG", "BND", "VTI", "VTSAX"]
    ticker_list = [t for t in ticker_list if t in safe_etfs]

elif risk_level == "Moderate risk (Stocks & ETFs)":
    avoid_high_risk = ["GME", "AMC", "MEME"]
    ticker_list = [t for t in ticker_list if t not in avoid_high_risk]

# High risk: no filtering
```

***

## 🔧 Troubleshooting

### Common Issues

#### 1. **Model Not Found Error**
```
❌ Error: Model not found at: models/ppo_dynamic_portfolio_padded
```

**Solution:**
```bash
# Verify models directory exists
ls -la models/

# Download model from Hugging Face or GitHub releases
# Or train a new model using training notebooks
```

#### 2. **Insufficient Data**
```
❌ Error: Insufficient data. Try different tickers or longer lookback period.
```

**Solution:**
```bash
# Increase lookback period in sidebar (e.g., 180 days instead of 90)
# Use more liquid tickers (AAPL, MSFT, GOOGL vs. penny stocks)
# Verify Yahoo Finance connection
```

#### 3. **Port Already in Use**
```
❌ Error: Address already in use 127.0.0.1:8501
```

**Solution:**
```bash
# Use different port
streamlit run streamlit_paper_trading.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9  # macOS/Linux
```

#### 4. **Out of Memory**
```
❌ Error: CUDA out of memory / MemoryError
```

**Solution:**
```bash
# Reduce lookback period
# Use fewer assets
# Increase system RAM or use CPU-only inference
```

### Debugging Mode

```python
# Add to streamlit_paper_trading.py for verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# All steps will print debug info
```

***

## 🚀 Future Enhancements

### Planned Features (v2.0)
- [ ] Multi-model ensemble (PPO + SAC + DQN)
- [ ] Real-time live trading integration (Alpaca API)
- [ ] Portfolio optimization (HRP, Black-Litterman)
- [ ] Options strategy support
- [ ] Multi-timeframe analysis (1H, 4H, daily)
- [ ] Custom RL training pipeline
- [ ] Cloud deployment (AWS Lambda, GCP)
- [ ] Mobile app (React Native)

### Research Areas
- [ ] Transformer-based policy networks
- [ ] Meta-learning for rapid adaptation
- [ ] Adversarial robustness
- [ ] Explainable AI (SHAP, LIME)
- [ ] Causal inference for trading

***

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup
```bash
# Clone and setup
git clone https://github.com/aryamansingh01/IntelliWealth-AI-Powered-Portfolio.git
cd IntelliWealth-AI-Powered-Portfolio
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
pytest tests/

# Commit with descriptive message
git commit -m "feat: Add new feature description"

# Push and create pull request
git push origin feature/your-feature-name
```

### Coding Standards
- Use **Black** for code formatting: `black streamlit_paper_trading.py`
- Follow **PEP 8** style guide
- Add type hints: `def calculate_return(portfolio: Dict[str, float]) -> float:`
- Include docstrings: `"""Function description and usage."""`

***

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Summary:** You're free to use, modify, and distribute this software, including commercially, as long as you include the original license.

***

## 👤 Author

**Aryaman Singh**
- Email: aryamansingh585@gmail.com
- GitHub: [@aryamansingh01](https://github.com/aryamansingh01)
- LinkedIn: [Aryaman Singh](https://www.linkedin.com/in/aryaman-singh)

***

## 🙏 Acknowledgments

- **Stable-Baselines3** - RL algorithm implementations
- **Streamlit** - Amazing web framework
- **Yahoo Finance** - Market data API
- **OpenAI Gym** - Environment standard
- **PyTorch** - Deep learning framework

***

## 📞 Support

Have questions? Here's where to get help:

1. **Documentation**: Check [docs/](docs/) folder
2. **GitHub Issues**: [Report bugs](https://github.com/aryamansingh01/IntelliWealth-AI-Powered-Portfolio/issues)
3. **Email**: aryamansingh@email.com
4. **LinkedIn**: Connect for professional inquiries

***

## 📈 Star History

If this project helped you, please consider giving it a ⭐!

```
⭐⭐⭐⭐⭐ Thank you for using IntelliWealth!
```
