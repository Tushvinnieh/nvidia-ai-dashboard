# app.py - NVIDIA AI Dashboard (COMPLETE VERSION)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import warnings
#import openai
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="NVIDIA AI Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state for all features
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'openai_key' not in st.session_state:
    st.session_state.openai_key = None
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'NVDA': {'shares': 10, 'avg_price': 450.00},
        'AMD': {'shares': 20, 'avg_price': 120.00},
        'INTC': {'shares': 30, 'avg_price': 40.00}
    }

# Title
st.title("🚀 NVIDIA AI Dashboard")
st.markdown("AI-powered analytics for NVIDIA and portfolio management")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a task:",
    [
        "📈 Stock Analysis", 
        "📊 Technical Indicators",  # NEW
        "🏦 Portfolio Tracker",     # NEW
        "🤖 AI Text Summarization", 
        "💬 AI Chat Assistant",     # NEW
        "📰 Market Intelligence"
    ]
)

# Function to get REAL NVIDIA stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker="NVDA", days=30):
    """Fetch stock data from Yahoo Finance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if not hist.empty:
            return hist
    except Exception as e:
        st.warning(f"API Error for {ticker}: {e}")
    return None

def generate_sample_data(ticker="NVDA", days=30):
    """Generate realistic sample data if API fails"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Different base prices for different stocks
    base_prices = {
        'NVDA': 450,
        'AMD': 120,
        'INTC': 40,
        'MSFT': 300,
        'AAPL': 170
    }
    
    base_price = base_prices.get(ticker, 100)
    returns = np.random.normal(0.001, 0.02, days)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, days)),
        'High': prices * (1 + np.abs(np.random.normal(0.01, 0.01, days))),
        'Low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, days))),
        'Close': prices,
        'Volume': np.random.randint(20000000, 80000000, days)
    }, index=dates)
    
    return data

# ============ TECHNICAL INDICATORS FUNCTIONS ============
def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, period=20):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, sma, lower_band

# ============ PORTFOLIO FUNCTIONS ============
def calculate_portfolio_value():
    """Calculate current portfolio value and performance"""
    portfolio_value = 0
    total_investment = 0
    performance_data = []
    
    for ticker, details in st.session_state.portfolio.items():
        shares = details['shares']
        avg_price = details['avg_price']
        
        # Get current price
        data = get_stock_data(ticker, 1)
        if data is not None and len(data) > 0:
            current_price = data['Close'].iloc[-1]
        else:
            current_price = avg_price * 1.05  # Simulate 5% gain
        
        current_value = shares * current_price
        investment = shares * avg_price
        gain = current_value - investment
        gain_pct = (gain / investment) * 100 if investment > 0 else 0
        
        portfolio_value += current_value
        total_investment += investment
        
        performance_data.append({
            'Ticker': ticker,
            'Shares': shares,
            'Avg Price': f"${avg_price:.2f}",
            'Current Price': f"${current_price:.2f}",
            'Current Value': f"${current_value:.2f}",
            'Gain/Loss': f"${gain:.2f}",
            'Gain %': f"{gain_pct:.1f}%"
        })
    
    total_gain = portfolio_value - total_investment
    total_gain_pct = (total_gain / total_investment) * 100 if total_investment > 0 else 0
    
    return portfolio_value, total_investment, total_gain, total_gain_pct, performance_data

# ============ AI CHAT FUNCTIONS ============
def chat_with_nvidia_ai(user_message, api_key=None):
    """Chat with AI about NVIDIA topics"""
    
    if not api_key:
        return get_simulated_response(user_message)
    
    try:
        openai.api_key = api_key
        
        system_prompt = """You are an AI assistant specialized in NVIDIA Corporation and stock market analysis. 
        Provide accurate, helpful information about NVIDIA stock, products, earnings, and competitors.
        Be concise and data-driven."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]],
            {"role": "user", "content": user_message}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"OpenAI Error: {str(e)[:100]}")
        return get_simulated_response(user_message)

def get_simulated_response(user_message):
    """Simulated AI responses when no API key"""
    user_lower = user_message.lower()
    
    if any(word in user_lower for word in ['price', 'stock', 'trading', 'nvda']):
        return "💰 **NVIDIA (NVDA)**\nCurrent: ~$500-550 range\n52-Week High: ~$700\n52-Week Low: ~$280\nTrend: Strong AI demand driving growth"
    
    elif any(word in user_lower for word in ['earnings', 'revenue', 'profit']):
        return "📊 **Latest Earnings (Q4 2023)**\n• Revenue: $22.1B (+265% YoY)\n• Data Center: $18.4B\n• Gaming: $2.9B\n• Net Income: $12.3B\n• Next Earnings: May 2024"
    
    elif any(word in user_lower for word in ['technical', 'rsi', 'macd', 'indicator']):
        return "📈 **Technical Analysis**\n• RSI: ~60 (Neutral)\n• MACD: Bullish crossover\n• Support: $480\n• Resistance: $550\n• Trend: Upward channel"
    
    elif any(word in user_lower for word in ['portfolio', 'invest', 'holdings']):
        return "🏦 **Portfolio Management Tips**\n1. Diversify across sectors\n2. NVIDIA: Core holding for AI exposure\n3. AMD: Complementary position\n4. Rebalance quarterly\n5. Consider risk tolerance"
    
    else:
        return "I'm your NVIDIA AI assistant! Ask me about:\n• Stock prices and analysis\n• Technical indicators\n• Portfolio management\n• Earnings and financials\n• Products and competition"

# ============ MAIN APP LOGIC ============
if app_mode == "📈 Stock Analysis":
    st.header("📈 NVIDIA Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        days = st.slider("Select time period:", 7, 365, 60)
        
        if st.button("📊 Fetch Stock Data", type="primary"):
            with st.spinner("Loading data..."):
                # FIX 1: Line 209 - Changed from 'or' to proper check
                data = get_stock_data("NVDA", days)
                if data is None or data.empty:
                    data = generate_sample_data("NVDA", days)
                st.session_state.stock_data = data
                
                if data is not None:
                    fig = go.Figure(data=[go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='NVDA',
                        increasing_line_color='#00ff00',
                        decreasing_line_color='#ff0000'
                    )])
                    
                    fig.update_layout(
                        title=f"💰 NVIDIA (NVDA) - Last {days} Days",
                        yaxis_title="Price (USD)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Live Metrics")
        if st.session_state.stock_data is not None:
            data = st.session_state.stock_data
            latest = data.iloc[-1]
            
            st.metric("Current Price", f"${latest['Close']:.2f}")
            st.metric("Day High", f"${latest['High']:.2f}")
            st.metric("Day Low", f"${latest['Low']:.2f}")
            st.metric("Volume", f"{latest['Volume']:,}")

elif app_mode == "📊 Technical Indicators":
    st.header("📊 Advanced Technical Analysis")
    
    ticker = st.selectbox("Select Stock:", ["NVDA", "AMD", "INTC", "MSFT", "AAPL"])
    days = st.slider("Analysis Period (days):", 30, 365, 90)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Analysis", type="primary"):
            with st.spinner("Calculating technical indicators..."):
                # FIX 2: Line 274 - Changed from 'or' to proper check
                data = get_stock_data(ticker, days)
                if data is None or data.empty:
                    data = generate_sample_data(ticker, days)
                
                if data is not None and not data.empty:
                    # Create subplots
                    fig = make_subplots(
                        rows=4, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Price with Bollinger Bands', 'RSI (14)', 'MACD', 'Volume'),
                        row_heights=[0.4, 0.2, 0.2, 0.2]
                    )
                    
                    # Price with Bollinger Bands
                    upper_band, middle_band, lower_band = calculate_bollinger_bands(data)
                    
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=upper_band, name='Upper BB', line=dict(color='gray', dash='dash')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=middle_band, name='Middle BB', line=dict(color='gray')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=lower_band, name='Lower BB', line=dict(color='gray', dash='dash'), fill='tonexty'), row=1, col=1)
                    
                    # RSI
                    rsi = calculate_rsi(data)
                    fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='orange')), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    # MACD
                    macd, signal, hist = calculate_macd(data)
                    fig.add_trace(go.Scatter(x=data.index, y=macd, name='MACD', line=dict(color='blue')), row=3, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=signal, name='Signal', line=dict(color='red')), row=3, col=1)
                    fig.add_trace(go.Bar(x=data.index, y=hist, name='Histogram'), row=3, col=1)
                    
                    # Volume
                    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'), row=4, col=1)
                    
                    fig.update_layout(height=800, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Indicator Signals")
        
        if 'data' in locals() and data is not None:
            # Calculate current values
            current_rsi = calculate_rsi(data).iloc[-1]
            macd_line, signal_line, _ = calculate_macd(data)
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            
            # RSI Analysis
            st.markdown("### RSI (14)")
            st.metric("Current RSI", f"{current_rsi:.1f}")
            if current_rsi > 70:
                st.error("⚡ **Overbought** - Consider taking profits")
            elif current_rsi < 30:
                st.success("💎 **Oversold** - Potential buying opportunity")
            else:
                st.info("⚖️ **Neutral** - No extreme conditions")
            
            # MACD Analysis
            st.markdown("### MACD")
            st.metric("MACD Line", f"{current_macd:.2f}")
            st.metric("Signal Line", f"{current_signal:.2f}")
            
            if current_macd > current_signal:
                st.success("📈 **Bullish** - MACD above Signal")
            else:
                st.warning("📉 **Bearish** - MACD below Signal")
            
            # Bollinger Bands Analysis
            upper, middle, lower = calculate_bollinger_bands(data)
            current_price = data['Close'].iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            
            st.markdown("### Bollinger Bands")
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("Upper Band", f"${current_upper:.2f}")
            st.metric("Lower Band", f"${current_lower:.2f}")
            
            if current_price > current_upper:
                st.error("🔴 **Above Upper Band** - Overbought")
            elif current_price < current_lower:
                st.success("🟢 **Below Lower Band** - Oversold")
            else:
                st.info("🟡 **Within Bands** - Normal range")

elif app_mode == "🏦 Portfolio Tracker":
    st.header("🏦 Personal Portfolio Tracker")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Portfolio Overview")
        
        # Calculate portfolio metrics
        total_value, total_invested, total_gain, gain_pct, performance = calculate_portfolio_value()
        
        # Display portfolio metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Value", f"${total_value:,.2f}")
        with metric_col2:
            st.metric("Total Invested", f"${total_invested:,.2f}")
        with metric_col3:
            st.metric("Total Gain/Loss", f"${total_gain:,.2f}", f"{gain_pct:.1f}%")
        
        # Performance chart
        st.subheader("📈 Portfolio Composition")
        
        if performance:
            df_performance = pd.DataFrame(performance)
            st.dataframe(df_performance, use_container_width=True)
            
            # Pie chart of holdings
            holdings = {p['Ticker']: float(p['Current Value'].replace('$', '').replace(',', '')) 
                       for p in performance}
            
            fig = go.Figure(data=[go.Pie(
                labels=list(holdings.keys()),
                values=list(holdings.values()),
                hole=0.3
            )])
            fig.update_layout(title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔄 Manage Portfolio")
        
        # Add new stock
        with st.expander("➕ Add Stock"):
            new_ticker = st.text_input("Ticker Symbol:", "AAPL").upper()
            new_shares = st.number_input("Number of shares:", min_value=1, value=10)
            new_price = st.number_input("Average price per share:", min_value=0.01, value=100.00)
            
            if st.button("Add to Portfolio"):
                if new_ticker not in st.session_state.portfolio:
                    st.session_state.portfolio[new_ticker] = {
                        'shares': new_shares,
                        'avg_price': new_price
                    }
                    st.success(f"Added {new_shares} shares of {new_ticker}")
                    st.rerun()
        
        # Remove stock
        with st.expander("➖ Remove Stock"):
            if st.session_state.portfolio:
                ticker_to_remove = st.selectbox("Select stock to remove:", 
                                               list(st.session_state.portfolio.keys()))
                if st.button("Remove from Portfolio"):
                    del st.session_state.portfolio[ticker_to_remove]
                    st.success(f"Removed {ticker_to_remove}")
                    st.rerun()
        
        # Edit existing stock
        with st.expander("✏️ Edit Holdings"):
            if st.session_state.portfolio:
                ticker_to_edit = st.selectbox("Select stock to edit:", 
                                             list(st.session_state.portfolio.keys()))
                current_shares = st.session_state.portfolio[ticker_to_edit]['shares']
                current_price = st.session_state.portfolio[ticker_to_edit]['avg_price']
                
                new_shares = st.number_input("New share count:", 
                                           min_value=0, 
                                           value=current_shares,
                                           key="edit_shares")
                new_price = st.number_input("New average price:", 
                                          min_value=0.01, 
                                          value=current_price,
                                          key="edit_price")
                
                if st.button("Update Holdings"):
                    st.session_state.portfolio[ticker_to_edit] = {
                        'shares': new_shares,
                        'avg_price': new_price
                    }
                    st.success(f"Updated {ticker_to_edit}")
                    st.rerun()
        
        # Portfolio actions
        st.markdown("---")
        if st.button("📥 Export Portfolio"):
            portfolio_df = pd.DataFrame([
                {
                    'Ticker': ticker,
                    'Shares': details['shares'],
                    'Avg Price': details['avg_price']
                }
                for ticker, details in st.session_state.portfolio.items()
            ])
            st.download_button(
                "Download CSV",
                portfolio_df.to_csv(index=False),
                "portfolio.csv",
                "text/csv"
            )
        
        if st.button("🔄 Reset Portfolio"):
            st.session_state.portfolio = {}
            st.rerun()

elif app_mode == "💬 AI Chat Assistant":
    st.header("💬 NVIDIA AI Chat Assistant")
    
    # API Key input in sidebar
    with st.sidebar.expander("🔑 OpenAI API Key (Optional)"):
        api_key = st.text_input("Enter your OpenAI key:", type="password")
        if api_key:
            st.session_state.openai_key = api_key
            st.success("✅ API key saved")
        else:
            st.info("Without key: Simulated responses")
            st.caption("[Get OpenAI API key](https://platform.openai.com/api-keys)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Chat with NVIDIA AI")
        st.caption("Ask about stocks, portfolio, technical analysis, or investment advice")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about NVIDIA or investing..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    response = chat_with_nvidia_ai(prompt, st.session_state.openai_key)
                    st.markdown(response)
            
            # Add AI response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.markdown("### 💡 Quick Questions")
        
        quick_questions = [
            "NVIDIA stock analysis?",
            "Technical indicators for NVDA?",
            "How to diversify portfolio?",
            "NVIDIA vs AMD comparison",
            "Best time to buy stocks?",
            "Portfolio rebalancing tips",
            "RSI trading strategy"
        ]
        
        for q in quick_questions:
            if st.button(f"❔ {q}", key=f"btn_{q[:10]}"):
                st.session_state.messages.append({"role": "user", "content": q})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(q)
                    with st.chat_message("assistant"):
                        with st.spinner("🤖 Thinking..."):
                            response = chat_with_nvidia_ai(q, st.session_state.openai_key)
                            st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.markdown("---")
        
        # Chat controls
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

elif app_mode == "🤖 AI Text Summarization":
    st.header("🤖 AI Text Summarization")
    
    st.info("Note: Transformers library commented out. Uncomment line 7 and install transformers for real AI.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_text = st.text_area("Paste your text:", height=300,
                                 placeholder="Paste article, report, or document here...")
        
        if st.button("Generate Summary") and input_text:
            # Simple summary (replace with AI when transformers installed)
            sentences = input_text.split('. ')
            if len(sentences) > 3:
                summary = '. '.join(sentences[:3]) + '.'
            else:
                summary = input_text
            st.session_state.summary = summary
    
    with col2:
        if st.session_state.summary:
            st.text_area("Summary:", value=st.session_state.summary, height=250)

elif app_mode == "📰 Market Intelligence":
    st.header("📰 Market Intelligence")
    
    # Compare multiple stocks
    st.subheader("📊 Multi-Stock Comparison")
    
    selected_stocks = st.multiselect(
        "Select stocks to compare:",
        ["NVDA", "AMD", "INTC", "MSFT", "AAPL", "GOOGL", "TSLA"],
        default=["NVDA", "AMD", "INTC"]
    )
    
    if selected_stocks and st.button("Compare Performance"):
        comparison_data = []
        
        for ticker in selected_stocks:
            # FIX 3: Line 479 - Changed from 'or' to proper check
            data = get_stock_data(ticker, 30)
            if data is None or data.empty:
                data = generate_sample_data(ticker, 30)
                
            if data is not None and len(data) > 0:
                latest_price = data['Close'].iloc[-1]
                price_change = ((latest_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                volume = data['Volume'].mean()
                
                comparison_data.append({
                    'Ticker': ticker,
                    'Current Price': f"${latest_price:.2f}",
                    '30-Day Change': f"{price_change:+.1f}%",
                    'Avg Volume': f"{volume:,.0f}"
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)

