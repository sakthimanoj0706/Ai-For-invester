from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="google")

from src.db import get_recent_signals
from src.utils import format_score
from src.scoring import rank_signals

st.set_page_config(
    page_title="SignalAI",
    layout="wide"
)

st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0A1628;
    }
    
    /* Font Coloring */
    h1, h2, h3, p, span, div.stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* Metrics / Accents */
    [data-testid="stMetricValue"] {
        color: #F0A500 !important;
    }
    
    /* Card Emulation */
    .signal-card {
        background-color: #132338;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    
    /* Expander UI Fixes */
    .streamlit-expanderHeader {
        background-color: #132338 !important;
        color: white !important;
        border-radius: 6px;
    }
    
    /* Fix text input label visibility */
    .stTextInput label {
        color: #B8C7D9 !important;
        font-size: 0.9em !important;
        font-weight: 500 !important;
    }
    
    /* Fix text input border */
    .stTextInput input {
        background-color: #132338 !important;
        color: #FFFFFF !important;
        border: 1px solid #2C3E50 !important;
        border-radius: 6px !important;
    }
    
    /* Fix text input focus border */
    .stTextInput input:focus {
        border-color: #F0A500 !important;
        box-shadow: 0 0 0 1px #F0A500 !important;
    }
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = []
    try:
        recent = get_recent_signals(1)
        if recent and isinstance(recent, list):
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_signals = [r for r in recent if str(r.get("timestamp", "")).startswith(today_str)]
            if today_signals:
                for s in today_signals:
                    s["combined_score"] = s.pop("score", 0.0)
                    s['signals_triggered'] = [x.strip() for x in str(s.get('signals', '')).split(',')] if s.get('signals') else []
                    s["is_fallback"] = 'Weak Momentum' in s['signals_triggered']
                st.session_state.results = today_signals
    except Exception:
        pass

if "last_run_time" not in st.session_state:
    st.session_state["last_run_time"] = None

st.markdown("""
<div style='padding:20px 0 10px'>
  <h1 style='color:#FFFFFF;margin:0;font-size:2.2em'>
    📊 SignalAI
  </h1>
  <p style='color:#B8C7D9;margin:4px 0 0;font-size:1em'>
    AI-Powered Market Intelligence for Indian Investors
  </p>
</div>
""", unsafe_allow_html=True)

col_h, col_btn = st.columns([4, 1])

with col_h:
    holdings_input = st.text_input(
        "Your holdings",
        placeholder="TCS, RELIANCE, INFY...",
        key="holdings_box",
        label_visibility="collapsed"
    )
    if holdings_input:
        st.caption(f"Portfolio filter active: "
                   f"{holdings_input.upper()}")

with col_btn:
    st.markdown("<div style='margin-top:4px'>",
                unsafe_allow_html=True)
    
    if "pipeline_running" not in st.session_state:
        st.session_state["pipeline_running"] = False
    
    run_clicked = st.button(
        "🚀 Run Signals",
        key="run_pipeline_btn",
        use_container_width=True,
        disabled=st.session_state["pipeline_running"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

if run_clicked:
    st.session_state["pipeline_running"] = True
    with st.spinner("Running pipeline..."):
        try:
            from run_pipeline import run_pipeline
            results = run_pipeline()
            st.session_state["results"] = results
            st.session_state["last_run_time"] = (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            st.success(
                f"Pipeline complete! "
                f"{len(results)} signals detected."
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
        finally:
            st.session_state["pipeline_running"] = False
            st.rerun()

if st.session_state.get("last_run_time"):
    st.caption(
        f"Last updated: "
        f"{st.session_state['last_run_time']}"
    )

st.markdown("---")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

results = st.session_state.get("results", [])

if results:
    st.markdown(
        "<h3 style='color:#FFFFFF;margin-bottom:16px'>"
        "📈 Today's Top 3 Signals</h3>",
        unsafe_allow_html=True
    )
    
    # Load universe for chart data
    chart_cols = st.columns(3)
    
    for idx, signal in enumerate(results[:3]):
        sym = signal.get("symbol", "")
        score = signal.get("combined_score", 0)
        signals_list = signal.get(
            "signals_triggered", [])
        close = signal.get("close", 0)
        change = signal.get("price_1d_change", 0)
        
        with chart_cols[idx]:
            # Score color
            if score >= 6:
                s_color = "#1DB954"
            elif score >= 3.5:
                s_color = "#F0A500"
            elif score >= 2:
                s_color = "#3498db"
            else:
                s_color = "#8FA3BE"
            
            # Mini chart using matplotlib
            try:
                import yfinance as yf
                ticker = yf.Ticker(f"{sym}.NS")
                hist = ticker.history(period="30d")
                
                fig, ax = plt.subplots(
                    figsize=(4, 2.2),
                    facecolor="#132338"
                )
                ax.set_facecolor="#132338"
                
                if not hist.empty:
                    prices = hist["Close"].values
                    dates = range(len(prices))
                    
                    # Color line based on trend
                    line_color = (
                        "#1DB954" if prices[-1] > prices[0]
                        else "#E24B4B"
                    )
                    ax.plot(
                        dates, prices,
                        color=line_color,
                        linewidth=1.5
                    )
                    ax.fill_between(
                        dates, prices,
                        prices.min(),
                        alpha=0.15,
                        color=line_color
                    )
                    
                    # Mark today's price
                    ax.axhline(
                        y=prices[-1],
                        color="#F0A500",
                        linewidth=0.8,
                        linestyle="--",
                        alpha=0.7
                    )
                
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                plt.tight_layout(pad=0.1)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
            except Exception:
                st.markdown(
                    f"<div style='background:#132338;"
                    f"height:88px;border-radius:8px;"
                    f"display:flex;align-items:center;"
                    f"justify-content:center;"
                    f"color:#8FA3BE;font-size:0.8em'>"
                    f"Chart loading...</div>",
                    unsafe_allow_html=True
                )
            
            # Stock name + score pill
            change_color = (
                "#1DB954" if change >= 0
                else "#E24B4B"
            )
            change_arrow = "▲" if change >= 0 else "▼"
            
            st.markdown(f"""
            <div style='background:#132338;
                        border-radius:8px;
                        padding:10px 12px;
                        border:1px solid #1E3A5F;
                        margin-top:6px'>
                <div style='display:flex;
                            justify-content:space-between;
                            align-items:center'>
                    <span style='color:#FFFFFF;
                                 font-weight:bold;
                                 font-size:1em'>
                        {sym}
                    </span>
                    <span style='background:{s_color};
                                 color:#000;
                                 padding:2px 8px;
                                 border-radius:12px;
                                 font-size:0.75em;
                                 font-weight:bold'>
                        {score:.1f}
                    </span>
                </div>
                <div style='color:#B8C7D9;
                            font-size:0.82em;
                            margin-top:4px'>
                    ₹{close:,.2f}
                    <span style='color:{change_color};
                                 margin-left:8px'>
                        {change_arrow} {abs(change):.2f}%
                    </span>
                </div>
                <div style='margin-top:6px'>
                    {" ".join([
                        f"<span style='background:#1E3A5F;"
                        f"color:#B8C7D9;padding:2px 6px;"
                        f"border-radius:4px;font-size:0.72em;"
                        f"margin-right:3px'>{s}</span>"
                        for s in signals_list[:2]
                    ])}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")

    if results:
        # Portfolio reordering
        active_results = list(results)
        if holdings_input:
            portfolio_syms = [
                s.strip().upper()
                for s in holdings_input.split(",")
            ]
            held = [r for r in active_results
                    if r.get("symbol","") in portfolio_syms]
            unheld = [r for r in active_results
                      if r.get("symbol","") not in portfolio_syms]
            active_results = held + unheld
            
            no_signal = [
                s for s in portfolio_syms
                if s not in [r.get("symbol","")
                             for r in active_results]
            ]
            if no_signal:
                st.info(
                    f"📭 No signals today for: "
                    f"{', '.join(no_signal)}"
                )
        
        # Render top 3 cards
        for idx, r in enumerate(active_results[:3]):
            sym = r.get("symbol", "Unknown")
            score = r.get("combined_score", 0.0)
            is_held = (holdings_input and sym in [
                s.strip().upper()
                for s in holdings_input.split(",")
            ])
            is_fallback = r.get("is_fallback", False)
            
            st.markdown(
                '<div class="signal-card">',
                unsafe_allow_html=True
            )
            
            # Top label
            if idx == 0:
                label = (
                    "📌 Your Portfolio Stock"
                    if is_held
                    else "🌟 Top Opportunity Today"
                )
                label_color = (
                    "#F0A500" if is_held
                    else "#1DB954"
                )
                st.markdown(
                    f"<h4 style='color:{label_color};"
                    f"margin-bottom:8px'>{label}</h4>",
                    unsafe_allow_html=True
                )
            
            # Header row
            c1, c2 = st.columns([3, 1])
            with c1:
                portfolio_badge = (
                    " 📌 In Your Portfolio"
                    if is_held else ""
                )
                st.markdown(
                    f"## {sym}{portfolio_badge}"
                )
                if is_fallback:
                    st.markdown(
                        "<span style='background:#F39C12;"
                        "color:white;padding:4px 8px;"
                        "border-radius:4px;"
                        "font-size:0.85em;"
                        "font-weight:bold'>"
                        "Early Momentum Watch</span>",
                        unsafe_allow_html=True
                    )
            
            with c2:
                score_color = (
                    "#1DB954" if score >= 6
                    else "#F0A500" if score >= 3.5
                    else "#8FA3BE"
                )
                st.markdown(
                    f"<h3 style='color:{score_color};"
                    f"text-align:right'>"
                    f"{score:.1f} / 10</h3>",
                    unsafe_allow_html=True
                )
                
                # Key metric
                vol = r.get("volume_ratio", 0)
                rsi = r.get("rsi_today", 0)
                chg = r.get("price_1d_change", 0)
                
                if vol and float(vol) > 1.5:
                    metric_txt = f"Vol: {float(vol):.1f}x"
                    metric_color = "#1DB954"
                elif rsi:
                    metric_txt = f"RSI: {float(rsi):.0f}"
                    metric_color = "#F0A500"
                else:
                    metric_txt = f"Chg: {float(chg):+.2f}%"
                    metric_color = "#8FA3BE"
                
                st.markdown(
                    f"<p style='text-align:right;"
                    f"color:{metric_color};"
                    f"font-size:0.9em'>"
                    f"{metric_txt}</p>",
                    unsafe_allow_html=True
                )
            
            # Signal badges
            triggers = r.get("signals_triggered", [])
            confidence = r.get("confidence", "Low")
            conf_color = (
                "#1DB954" if confidence == "High"
                else "#F0A500" if confidence in
                     ["Medium", "Watch"]
                else "#E24B4B"
            )
            
            badge_html = "".join([
                f"<span style='background:#1E3A5F;"
                f"color:#ECF0F1;padding:4px 10px;"
                f"border-radius:4px;font-size:0.85em;"
                f"margin-right:6px'>{t}</span>"
                for t in triggers
            ])
            conf_badge = (
                f"<span style='border:1px solid "
                f"{conf_color};color:{conf_color};"
                f"padding:3px 10px;border-radius:12px;"
                f"font-size:0.85em'>"
                f"{confidence} Confidence</span>"
            )
            st.markdown(
                f"<div style='margin:10px 0'>"
                f"{badge_html}{conf_badge}</div>",
                unsafe_allow_html=True
            )
            
            # Reasoning
            reasoning = r.get("agent_reasoning", "")
            if reasoning:
                st.markdown(
                    f"<p style='color:#B8C7D9;"
                    f"font-size:0.95em'>"
                    f"<strong>Why:</strong> "
                    f"{reasoning}</p>",
                    unsafe_allow_html=True
                )
            
            # AI Explanation tabs
            with st.expander("🤖 AI Explanation"):
                exp_en = r.get("explanation_en", "")
                if exp_en and len(exp_en) > 40:
                    tab1, tab2, tab3 = st.tabs([
                        "English", "தமிழ்", "हिन्दी"
                    ])
                    with tab1:
                        st.write(exp_en)
                    with tab2:
                        st.write(
                            r.get("explanation_ta",
                                  exp_en)
                        )
                    with tab3:
                        st.write(
                            r.get("explanation_hi",
                                  exp_en)
                        )
                else:
                    st.info(
                        "Run pipeline to generate "
                        "AI explanation"
                    )
            
            # Video
            import os
            vid_path = r.get("video_path", "")
            if vid_path and os.path.exists(str(vid_path)):
                st.video(str(vid_path))
            else:
                st.info(
                    f"Re-run pipeline to generate "
                    f"video for {sym}"
                )
            
            st.markdown(
                '</div>', unsafe_allow_html=True
            )
    
else:
    st.info(
        "Click '🚀 Run Signals' to generate "
        "today's market intelligence."
    )

st.markdown("---")
st.markdown(
    "<h3 style='color:#FFFFFF'>"
    "📰 Market Events</h3>",
    unsafe_allow_html=True
)

NEWS_EVENTS = [
    {
        "headline": "RBI cuts repo rate by 25bps to 6.25%",
        "type": "MACRO",
        "impact": "positive",
        "date": "2026-03-27",
        "sectors": "Banking · Auto"
    },
    {
        "headline": "Strong USD boosts Indian IT margins",
        "type": "SECTOR",
        "impact": "positive",
        "date": "2026-03-27",
        "sectors": "IT · Infra"
    },
    {
        "headline": "Crude oil falls 3% — positive for India",
        "type": "MACRO",
        "impact": "mixed",
        "date": "2026-03-27",
        "sectors": "Energy · Auto"
    },
    {
        "headline": "USFDA clears major Indian pharma plant",
        "type": "REGULATORY",
        "impact": "positive",
        "date": "2026-03-27",
        "sectors": "Pharma"
    },
]

news_cols = st.columns(2)
for i, event in enumerate(NEWS_EVENTS):
    impact = event.get("impact", "neutral")
    color = (
        "#1DB954" if impact == "positive"
        else "#E24B4B" if impact == "negative"
        else "#F0A500"
    )
    with news_cols[i % 2]:
        st.markdown(f"""
        <div style='background:#132338;
                    padding:12px 14px;
                    border-radius:8px;
                    border-left:3px solid {color};
                    margin-bottom:10px'>
            <div style='color:{color};
                        font-size:0.72em;
                        font-weight:bold;
                        margin-bottom:4px'>
                {event["type"]} · {impact.upper()}
            </div>
            <div style='color:#FFFFFF;
                        font-size:0.92em;
                        font-weight:500'>
                {event["headline"]}
            </div>
            <div style='color:#8FA3BE;
                        font-size:0.78em;
                        margin-top:5px'>
                {event["sectors"]} · {event["date"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
with st.expander("🔍 Search Any Stock", expanded=False):
    s_col1, s_col2 = st.columns([4, 1])
    with s_col1:
        search_sym = st.text_input(
            "Stock symbol",
            placeholder="TCS, RELIANCE, INFY...",
            key="search_input",
            label_visibility="collapsed"
        ).strip().upper()
    with s_col2:
        search_btn = st.button(
            "Search", key="search_btn"
        )
    
    UNIVERSE = [
        "RELIANCE","TCS","HDFCBANK","INFY",
        "ICICIBANK","MARUTI","BAJFINANCE",
        "SUNPHARMA","DRREDDY","CIPLA",
        "ADANIENT","NTPC","POWERGRID","WIPRO","M&M"
    ]
    
    if search_sym:
        results_now = st.session_state.get(
            "results", [])
        match = next(
            (r for r in results_now
             if r.get("symbol","").upper() == search_sym),
            None
        )
        
        if match:
            st.success(
                f"✅ {search_sym} found in today's analysis"
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Score",
                f"{match.get('combined_score',0):.1f}/10"
            )
            m2.metric(
                "RSI",
                f"{match.get('rsi_today',0):.0f}"
            )
            m3.metric(
                "Volume",
                f"{match.get('volume_ratio',0):.1f}x"
            )
            m4.metric(
                "1D Change",
                f"{match.get('price_1d_change',0):+.2f}%"
            )
            
            signals_list = match.get(
                "signals_triggered", [])
            if signals_list:
                st.markdown(
                    "**Signals:** " +
                    " · ".join(signals_list)
                )
            
            st.markdown(
                f"**Why:** "
                f"{match.get('agent_reasoning','')}"
            )
            
            vid = match.get("video_path", "")
            if vid and os.path.exists(str(vid)):
                st.video(str(vid))
                
        elif search_sym in UNIVERSE:
            st.warning(
                f"⚠ {search_sym} is tracked but had "
                f"no strong signal today. "
                f"Run pipeline for latest data."
            )
            
            # Show live price anyway
            try:
                import yfinance as yf
                ticker = yf.Ticker(f"{search_sym}.NS")
                hist = ticker.history(period="5d")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
                    prev = hist["Close"].iloc[-2]
                    chg = ((price-prev)/prev)*100
                    st.metric(
                        f"{search_sym} Current Price",
                        f"₹{price:,.2f}",
                        f"{chg:+.2f}%"
                    )
            except Exception:
                pass
        else:
            st.error(
                f"❌ {search_sym} not in our universe. "
                f"Covered: {', '.join(UNIVERSE)}"
            )

st.markdown("---")
st.markdown(
    "<h3 style='color:#FFFFFF'>"
    "📋 Activity Ledger</h3>",
    unsafe_allow_html=True
)

try:
    from src.db import get_recent_signals
    logs = get_recent_signals(7)
    if logs:
        df_logs = pd.DataFrame(logs)
        drop_cols = [
            "explanation_en", "explanation_ta",
            "explanation_hi"
        ]
        display_df = df_logs.drop(
            columns=drop_cols, errors="ignore"
        )
        st.dataframe(
            display_df,
            width="stretch"
        )
        csv_data = df_logs.to_csv(
            index=False
        ).encode("utf-8")
        st.download_button(
            "⬇ Download CSV",
            data=csv_data,
            file_name="signalai_logs.csv",
            mime="text/csv"
        )
    else:
        st.info("No logs yet. Run pipeline first.")
except Exception as e:
    st.error(f"Log error: {e}")
