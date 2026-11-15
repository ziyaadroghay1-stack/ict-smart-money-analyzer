import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="ICT Smart Money Analyzer", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp { background: #0b1220; color: #e5e7eb; }
[data-testid="stMetric"] { background: #131f32; border: 1px solid #26364d; padding: 14px; border-radius: 12px; }
section[data-testid="stSidebar"] { background: #101a2b; }
div[data-testid="stDataFrame"] { border: 1px solid #26364d; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 ICT SMART MONEY ANALYZER")
st.caption("Educational market-structure simulator • NAS100 & US30 • No broker connection")

st.warning(
    "SIMULATION / EDUCATION ONLY. This app identifies simplified chart patterns, "
    "not verified trading opportunities. It does not provide entries, targets, stop-losses, "
    "live execution, or financial advice."
)

# ---------- Data ----------
def make_demo_data(seed: int, periods: int = 1800) -> pd.DataFrame:
    """Create illustrative OHLC candles. These are not real market prices."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("5min"),
        periods=periods,
        freq="5min"
    )
    drift = rng.normal(0, 1.0, periods) + 0.025 * np.sin(np.arange(periods) / 65)
    close = 18000 + np.cumsum(drift)
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.15, 1.2, periods)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=idx
    )

def load_csv(uploaded) -> pd.DataFrame:
    raw = pd.read_csv(uploaded)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError("CSV must contain Open, High, Low and Close columns.")
    time_col = next((c for c in ["datetime", "timestamp", "date", "time"] if c in raw.columns), None)
    if time_col:
        idx = pd.to_datetime(raw[time_col], errors="coerce", utc=True)
        raw = raw.loc[idx.notna()].copy()
        raw.index = idx[idx.notna()]
    else:
        raw.index = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(raw), freq="5min")
        st.info("No date/time column found. A demonstration 5-minute timeline was assigned.")
    for c in required:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=required).sort_index()
    raw = raw.rename(columns={c: c.capitalize() for c in required})
    if raw.empty:
        raise ValueError("No valid OHLC rows were found in the CSV.")
    return raw[["Open", "High", "Low", "Close"]]

def analyse(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Causal confirmation: swing becomes usable only after right bars have closed."""
    x = df.copy().sort_index()
    # Basic candle validity
    valid = (
        (x.High >= x.Low) & (x.High >= x.Open) & (x.High >= x.Close) &
        (x.Low <= x.Open) & (x.Low <= x.Close)
    )
    x = x.loc[valid].copy()
    if len(x) < left + right + 10:
        raise ValueError("Not enough valid candles for analysis.")

    # Identify pivot at its original bar, then shift availability to confirmation bar.
    roll_hi = x.High.rolling(left + right + 1, center=True).max()
    roll_lo = x.Low.rolling(left + right + 1, center=True).min()
    pivot_hi = (x.High == roll_hi) & roll_hi.notna()
    pivot_lo = (x.Low == roll_lo) & roll_lo.notna()
    x["Confirmed_Swing_High"] = pivot_hi.shift(right).fillna(False).astype(bool)
    x["Confirmed_Swing_Low"] = pivot_lo.shift(right).fillna(False).astype(bool)

    high_level = x.High.where(x.Confirmed_Swing_High).ffill().shift(1)
    low_level = x.Low.where(x.Confirmed_Swing_Low).ffill().shift(1)
    x["Bullish_MSS"] = (x.Close > high_level) & (x.Close.shift(1) <= high_level)
    x["Bearish_MSS"] = (x.Close < low_level) & (x.Close.shift(1) >= low_level)
    x["Bullish_BOS"] = x["Bullish_MSS"]
    x["Bearish_BOS"] = x["Bearish_MSS"]

    # Three-candle imbalance screens.
    x["Bullish_FVG"] = x.Low > x.High.shift(2)
    x["Bearish_FVG"] = x.High < x.Low.shift(2)

    # Liquidity sweep approximation: wick takes prior N-bar extreme, close returns inside.
    prior_high = x.High.shift(1).rolling(20, min_periods=10).max()
    prior_low = x.Low.shift(1).rolling(20, min_periods=10).min()
    x["Buy_Side_Sweep"] = (x.High > prior_high) & (x.Close < prior_high)
    x["Sell_Side_Sweep"] = (x.Low < prior_low) & (x.Close > prior_low)

    # Session labels using timezone-aware conversion (timestamps are treated as UTC).
    utc_index = x.index
    if utc_index.tz is None:
        utc_index = utc_index.tz_localize("UTC")
    london = utc_index.tz_convert("Europe/London")
    new_york = utc_index.tz_convert("America/New_York")
    london_window = (london.hour >= 7) & (london.hour < 10)
    ny_window = (new_york.hour >= 8) & (new_york.hour < 11)
    x["Session"] = np.where(london_window, "London", np.where(ny_window, "New York", "Other"))

    recent_sweep = (x.Buy_Side_Sweep | x.Sell_Side_Sweep).rolling(5, min_periods=1).max().astype(bool)
    recent_mss = (x.Bullish_MSS | x.Bearish_MSS).rolling(5, min_periods=1).max().astype(bool)
    recent_fvg = (x.Bullish_FVG | x.Bearish_FVG).rolling(5, min_periods=1).max().astype(bool)

    x["ICT_2022_Screen"] = recent_sweep & recent_mss & recent_fvg
    x["Silver_Bullet_Screen"] = x.Session.isin(["London", "New York"]) & (x.Bullish_FVG | x.Bearish_FVG)
    x["Judas_Swing_Screen"] = x.Session.isin(["London", "New York"]) & (x.Buy_Side_Sweep | x.Sell_Side_Sweep)
    x["Liquidity_Screen"] = x.Buy_Side_Sweep | x.Sell_Side_Sweep
    x["Bullish_Pattern_Count"] = (
        x.Sell_Side_Sweep.astype(int) + x.Bullish_MSS.astype(int) + x.Bullish_FVG.astype(int)
    )
    x["Bearish_Pattern_Count"] = (
        x.Buy_Side_Sweep.astype(int) + x.Bearish_MSS.astype(int) + x.Bearish_FVG.astype(int)
    )
    return x

def bias_for(frame: pd.DataFrame) -> str:
    if len(frame) < 5:
        return "Insufficient data"
    change = frame.Close.iloc[-1] - frame.Close.iloc[0]
    if change > 0:
        return "Bullish structure"
    if change < 0:
        return "Bearish structure"
    return "Neutral"

def mtf_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, rule in [("4H", "4h"), ("1H", "1h"), ("15M", "15min"), ("5M", "5min")]:
        candles = df.resample(rule).agg(
            {"Open":"first", "High":"max", "Low":"min", "Close":"last"}
        ).dropna()
        rows.append({"Timeframe": label, "Candles": len(candles), "Structure view": bias_for(candles)})
    return pd.DataFrame(rows)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    market = st.selectbox("Market", ["NAS100", "US30"])
    source = st.radio("Data source", ["Demo candles", "Upload historical CSV"])
    uploaded = None
    if source == "Upload historical CSV":
        uploaded = st.file_uploader("Upload OHLC CSV", type=["csv"])
    st.caption("CSV columns required: Open, High, Low, Close. Optional time column: datetime, timestamp, date or time.")
    show_last = st.slider("Candles shown on chart", 100, 500, 200, 25)
    sound_enabled = st.checkbox("Enable simulated alert sound", value=True)

try:
    if source == "Upload historical CSV":
        if uploaded is None:
            st.info("Upload a historical OHLC CSV to begin, or select Demo candles.")
            st.stop()
        raw = load_csv(uploaded)
    else:
        raw = make_demo_data(42 if market == "NAS100" else 84)

    data = analyse(raw)
except Exception as exc:
    st.error(f"Could not prepare data: {exc}")
    st.stop()

# ---------- Dashboard ----------
latest = data.iloc[-1]
st.subheader(f"{market} • Market overview")
a, b, c, d = st.columns(4)
a.metric("Latest sample close", f"{latest.Close:,.2f}")
b.metric("Bullish FVGs", int(data.Bullish_FVG.sum()))
c.metric("Bearish FVGs", int(data.Bearish_FVG.sum()))
d.metric("Liquidity sweeps", int(data.Liquidity_Screen.sum()))

left_col, right_col = st.columns([1.35, 1])
with left_col:
    st.markdown("### Candlestick chart")
    chart_data = data.tail(show_last)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_data.index, open=chart_data.Open, high=chart_data.High,
        low=chart_data.Low, close=chart_data.Close, name=market
    ))
    for column, ycol, name, symbol in [
        ("Bullish_MSS", "Low", "Bullish MSS", "triangle-up"),
        ("Bearish_MSS", "High", "Bearish MSS", "triangle-down"),
        ("Bullish_FVG", "Low", "Bullish FVG", "circle"),
        ("Bearish_FVG", "High", "Bearish FVG", "x"),
        ("Buy_Side_Sweep", "High", "Buy-side sweep", "diamond"),
        ("Sell_Side_Sweep", "Low", "Sell-side sweep", "diamond"),
    ]:
        points = chart_data[chart_data[column]]
        if not points.empty:
            fig.add_trace(go.Scatter(
                x=points.index, y=points[ycol], mode="markers", name=name,
                marker={"size": 9, "symbol": symbol}
            ))
    fig.update_layout(
        template="plotly_dark", height=570, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown("### Multi-timeframe view")
    st.dataframe(mtf_summary(data), hide_index=True, use_container_width=True)
    st.markdown("### Pattern screens")
    pattern_cols = [
        ("ICT 2022 concept screen", "ICT_2022_Screen"),
        ("Silver Bullet time/FVG screen", "Silver_Bullet_Screen"),
        ("Judas Swing session/sweep screen", "Judas_Swing_Screen"),
        ("Liquidity sweep screen", "Liquidity_Screen"),
    ]
    for label, col in pattern_cols:
        count = int(data[col].sum())
        st.write(f"**{label}:** {count} detected")
    st.caption("These are simplified educational pattern screens, not complete ICT rule implementations.")

st.markdown("### Latest simulated pattern events")
event_cols = ["ICT_2022_Screen", "Silver_Bullet_Screen", "Judas_Swing_Screen", "Liquidity_Screen"]
events = data[data[event_cols].any(axis=1)].tail(12)
if events.empty:
    st.info("No pattern screens detected in the latest data window.")
else:
    display = events[["Close", "Session"] + event_cols].copy()
    display.index.name = "Timestamp (UTC)"
    st.dataframe(display, use_container_width=True)

    newest = events.iloc[-1]
    active = [name.replace("_Screen", "").replace("_", " ") for name in event_cols if bool(newest[name])]
    st.info("Latest educational pattern event: " + ", ".join(active))

    if sound_enabled:
        st.caption("Sound is played only after you press the button, because mobile browsers may block automatic audio.")
        if st.button("🔊 Play simulated alert sound"):
            st.components.v1.html("""
            <button id="beep" style="background:#24364f;color:white;border:0;padding:10px 16px;border-radius:8px">
            Playing alert…
            </button>
            <script>
            const Ctx = window.AudioContext || window.webkitAudioContext;
            const ctx = new Ctx();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine'; osc.frequency.value = 740;
            gain.gain.value = 0.12;
            osc.connect(gain); gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.22);
            </script>
            """, height=55)

st.markdown("### Data quality")
valid = (
    (data.High >= data.Low) & (data.High >= data.Open) &
    (data.High >= data.Close) & (data.Low <= data.Open) &
    (data.Low <= data.Close)
)
q1, q2, q3 = st.columns(3)
q1.metric("Candles analysed", len(data))
q2.metric("Valid candles", int(valid.sum()))
q3.metric("Missing OHLC values", int(data[["Open","High","Low","Close"]].isna().sum().sum()))

with st.expander("Method notes and limitations"):
    st.write("""
    - Demo data is randomly generated and does not represent actual NAS100 or US30 prices.
    - Upload historical OHLC CSV data for chart study. Timestamps are interpreted as UTC.
    - Swing points are made available only after the configured right-side confirmation candles.
    - MSS/BOS, FVG, sweeps and named ICT model screens are simplified approximations.
    - Session windows are illustrative and may need adjustment for a specific ICT definition.
    - This version counts pattern occurrences; it does not calculate trade returns or prove an edge.
    - No live market feed, broker integration, order placement, or external trade-alert delivery is included.
    """)
st.caption("ICT Smart Money Analyzer • Educational simulator • Version 1.0")
