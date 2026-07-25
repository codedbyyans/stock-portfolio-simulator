import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Monte Carlo Portfolio Simulator",
    page_icon="📈",
    layout="wide",
)

# ----------------------------------------------------------------------------
# SESSION STATE (visitor counter)
# ----------------------------------------------------------------------------
if "visitor_count" not in st.session_state:
    st.session_state.visitor_count = 1
else:
    st.session_state.visitor_count += 0  # keep as-is; incremented once below

if "counted" not in st.session_state:
    st.session_state.visitor_count += 1
    st.session_state.counted = True

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(tickers, years=5):
    """Download historical adjusted close prices for the given tickers."""
    end = datetime.today()
    start = end - timedelta(days=int(365.25 * years))
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        return None

    # Handle both single-ticker and multi-ticker column structures
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")
    return prices


def normalize_weights(raw_weights):
    total = sum(raw_weights)
    if total == 0:
        n = len(raw_weights)
        return [1 / n] * n
    return [w / total for w in raw_weights]


def run_monte_carlo(mean_returns, cov_matrix, weights, initial_investment,
                     years, num_simulations, trading_days=252):
    """
    Run a Monte Carlo simulation of portfolio value using
    correlated Geometric Brownian Motion across assets.
    """
    weights = np.array(weights)
    num_days = int(years * trading_days)
    num_assets = len(weights)

    # Cholesky decomposition for correlated random shocks
    try:
        chol = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Fall back to a small jitter if covariance matrix isn't positive definite
        jitter = np.eye(num_assets) * 1e-10
        chol = np.linalg.cholesky(cov_matrix + jitter)

    portfolio_paths = np.zeros((num_simulations, num_days + 1))
    portfolio_paths[:, 0] = initial_investment

    mean_returns = np.array(mean_returns)

    for sim in range(num_simulations):
        # Generate correlated daily returns for all assets at once
        random_shocks = np.random.normal(size=(num_days, num_assets))
        correlated_shocks = random_shocks @ chol.T
        daily_asset_returns = mean_returns + correlated_shocks

        # Combine asset returns into a single portfolio daily return using weights
        daily_portfolio_returns = daily_asset_returns @ weights

        # Compound daily returns into a portfolio value path
        growth_factors = np.cumprod(1 + daily_portfolio_returns)
        portfolio_paths[sim, 1:] = initial_investment * growth_factors

    return portfolio_paths


# ----------------------------------------------------------------------------
# SIDEBAR - USER INPUTS
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Portfolio Settings")

st.sidebar.markdown("### Stock Tickers")
tickers_input = st.sidebar.text_input(
    "Enter comma-separated tickers",
    value="SPY, QQQ",
    help="Example: SPY, QQQ, AAPL, VTI"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.sidebar.markdown("### Portfolio Weights")
st.sidebar.caption("Weights automatically normalize to 100%.")

raw_weights = []
if tickers:
    default_weight = round(100 / len(tickers), 1)
    for ticker in tickers:
        w = st.sidebar.slider(
            f"{ticker} weight (%)", min_value=0, max_value=100,
            value=int(default_weight), step=1, key=f"weight_{ticker}"
        )
        raw_weights.append(w)

    normalized_weights = normalize_weights(raw_weights)

    weight_display = ", ".join(
        [f"{t}: {w*100:.1f}%" for t, w in zip(tickers, normalized_weights)]
    )
    st.sidebar.caption(f"Normalized → {weight_display}")
else:
    normalized_weights = []

st.sidebar.markdown("### Investment Parameters")
initial_investment = st.sidebar.number_input(
    "Initial Investment ($)", min_value=100, max_value=100_000_000,
    value=10_000, step=500
)

time_horizon = st.sidebar.slider(
    "Time Horizon (years)", min_value=1, max_value=10, value=5, step=1
)

num_simulations = st.sidebar.slider(
    "Number of Simulations", min_value=100, max_value=1000, value=500, step=50
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Share This Tool")
share_url = "https://share.streamlit.io/your-username/your-repo-name/main/app.py"
st.sidebar.text_input(
    "Copy this link to share:",
    value=share_url,
    help="Replace this with your actual deployed Streamlit Cloud URL."
)

run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# MAIN PAGE
# ----------------------------------------------------------------------------
st.title("📈 Monte Carlo Portfolio Simulator")
st.markdown(
    "Estimate the range of future outcomes for your portfolio using "
    "historical data and Monte Carlo simulation with Geometric Brownian Motion."
)

if not tickers:
    st.warning("Please enter at least one valid ticker in the sidebar.")
    st.stop()

if run_button:
    with st.spinner("Fetching historical price data..."):
        try:
            prices = fetch_price_data(tickers, years=5)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    if prices is None or prices.empty:
        st.error(
            "No data was returned. Please check that your tickers are valid "
            "and try again."
        )
        st.stop()

    # Drop tickers that returned no data at all, and warn the user
    missing_tickers = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing_tickers:
        st.warning(
            f"The following tickers could not be found and were excluded: "
            f"{', '.join(missing_tickers)}"
        )
        valid_tickers = [t for t in tickers if t not in missing_tickers]
        if not valid_tickers:
            st.error("None of the entered tickers returned valid data. Please try different tickers.")
            st.stop()

        # Re-normalize weights over the remaining valid tickers
        keep_indices = [tickers.index(t) for t in valid_tickers]
        raw_weights = [raw_weights[i] for i in keep_indices]
        normalized_weights = normalize_weights(raw_weights)
        tickers = valid_tickers
        prices = prices[tickers]

    prices = prices.dropna()

    if len(prices) < 30:
        st.error(
            "Not enough overlapping historical data across these tickers "
            "to run a reliable simulation. Please try different tickers."
        )
        st.stop()

    # Daily returns and covariance matrix
    daily_returns = prices.pct_change().dropna()
    mean_returns = daily_returns.mean().values
    cov_matrix = daily_returns.cov().values

    with st.spinner(f"Running {num_simulations} Monte Carlo simulations..."):
        try:
            portfolio_paths = run_monte_carlo(
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                weights=normalized_weights,
                initial_investment=initial_investment,
                years=time_horizon,
                num_simulations=num_simulations,
            )
        except Exception as e:
            st.error(f"Simulation error: {e}")
            st.stop()

    final_values = portfolio_paths[:, -1]
    p10 = np.percentile(final_values, 10)
    p50 = np.percentile(final_values, 50)
    p90 = np.percentile(final_values, 90)
    prob_profit = float(np.mean(final_values > initial_investment)) * 100

    # ------------------------------------------------------------------
    # METRIC CARDS
    # ------------------------------------------------------------------
    st.markdown("### 📊 Projected Outcomes")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Initial Investment", f"${initial_investment:,.0f}")
    col2.metric("Worst Case (10th %ile)", f"${p10:,.0f}",
                delta=f"{(p10/initial_investment - 1) * 100:.1f}%")
    col3.metric("Median Case (50th %ile)", f"${p50:,.0f}",
                delta=f"{(p50/initial_investment - 1) * 100:.1f}%")
    col4.metric("Best Case (90th %ile)", f"${p90:,.0f}",
                delta=f"{(p90/initial_investment - 1) * 100:.1f}%")

    st.markdown(
        f"### 🎯 Probability of Ending in Profit: **{prob_profit:.1f}%**"
    )

    # ------------------------------------------------------------------
    # PLOTLY CHART
    # ------------------------------------------------------------------
    st.markdown("### 📉 Simulated Portfolio Paths")

    trading_days = 252
    num_days = int(time_horizon * trading_days)
    time_axis = np.linspace(0, time_horizon, num_days + 1)

    path_p10 = np.percentile(portfolio_paths, 10, axis=0)
    path_p50 = np.percentile(portfolio_paths, 50, axis=0)
    path_p90 = np.percentile(portfolio_paths, 90, axis=0)

    fig = go.Figure()

    # Plot a sample of individual paths lightly in the background
    sample_size = min(100, num_simulations)
    sample_indices = np.random.choice(num_simulations, sample_size, replace=False)
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=time_axis, y=portfolio_paths[idx],
            mode="lines",
            line=dict(width=0.5, color="rgba(100,100,255,0.08)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=time_axis, y=path_p90, mode="lines",
        line=dict(width=2, color="green", dash="dot"),
        name="90th Percentile (Best Case)"
    ))
    fig.add_trace(go.Scatter(
        x=time_axis, y=path_p50, mode="lines",
        line=dict(width=3, color="blue"),
        name="50th Percentile (Median)"
    ))
    fig.add_trace(go.Scatter(
        x=time_axis, y=path_p10, mode="lines",
        line=dict(width=2, color="red", dash="dot"),
        name="10th Percentile (Worst Case)"
    ))

    fig.update_layout(
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # DATA SUMMARY
    # ------------------------------------------------------------------
    with st.expander("📋 View Historical Data Summary"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Annualized Return (Historical)**")
            st.dataframe(
                (daily_returns.mean() * trading_days * 100).round(2).rename("Return (%)")
            )
        with col_b:
            st.markdown("**Annualized Volatility (Historical)**")
            st.dataframe(
                (daily_returns.std() * np.sqrt(trading_days) * 100).round(2).rename("Volatility (%)")
            )

else:
    st.info("Adjust your settings in the sidebar, then click **Run Simulation** to begin.")

# ----------------------------------------------------------------------------
# FOOTER - VISITOR COUNTER
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption(f"👁️ Page views this session: {st.session_state.visitor_count}")
st.caption(
    "⚠️ This tool is for educational purposes only and does not constitute "
    "financial advice. Past performance does not guarantee future results."
)
