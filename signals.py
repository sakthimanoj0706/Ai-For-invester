import pandas as pd
import numpy as np

def detect_signals(symbol: str,
                   df: pd.DataFrame) -> dict:
    """
    Detects 4 signal types with realistic thresholds.
    Returns enriched signal dict.
    """
    if df is None or len(df) < 30:
        return None

    # Get last row values
    latest = df.iloc[-1]
    close = float(latest["close"])
    volume = float(latest["volume"])
    open_price = float(latest["open"])
    high = float(latest["high"])

    # Indicators
    avg_vol_20 = df["volume"].rolling(20).mean().iloc[-1]
    avg_vol_20 = avg_vol_20 if avg_vol_20 > 0 else 1
    volume_ratio = volume / avg_vol_20

    # RSI
    if "rsi" in df.columns:
        rsi = float(df["rsi"].iloc[-1])
        rsi_series = df["rsi"]
    else:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1])

    if pd.isna(rsi):
        rsi = 50.0

    # Rolling highs for resistance
    rolling_20_high = df["high"].rolling(20).max().shift(1).iloc[-1]
    rolling_10_high = df["high"].rolling(10).max().shift(1).iloc[-1]
    rolling_5_high  = df["high"].rolling(5).max().shift(1).iloc[-1]

    green_candle = close > open_price

    # ── SIGNAL 1: Resistance Breakout ──────────
    # Tiered: 5-day > 10-day > 20-day breakout
    if close > rolling_20_high and volume_ratio > 1.2:
        breakout_triggered = True
        breakout_pct = ((close-rolling_20_high)
                        /rolling_20_high)*100
        s1_score = min(
            (breakout_pct/2)*4 + (volume_ratio/2)*6, 10)
        breakout_level = "20-day high"
    elif close > rolling_10_high and volume_ratio > 1.1:
        breakout_triggered = True
        breakout_pct = ((close-rolling_10_high)
                        /rolling_10_high)*100
        s1_score = min(
            (breakout_pct/2)*3 + (volume_ratio/2)*4, 7)
        breakout_level = "10-day high"
    elif close > rolling_5_high and volume_ratio > 1.0:
        breakout_triggered = True
        breakout_pct = ((close-rolling_5_high)
                        /rolling_5_high)*100
        s1_score = min(
            (breakout_pct/2)*2 + (volume_ratio/2)*3, 5)
        breakout_level = "5-day high"
    else:
        breakout_triggered = False
        breakout_pct = 0.0
        s1_score = 0.0
        breakout_level = "none"

    # ── SIGNAL 2: Volume Spike ──────────────────
    if volume_ratio > 1.8 and green_candle:
        volume_spike_triggered = True
        s2_score = min(volume_ratio * 1.5, 10)
    elif volume_ratio > 1.4 and green_candle:
        volume_spike_triggered = True
        s2_score = min(volume_ratio * 1.0, 6)
    else:
        volume_spike_triggered = False
        s2_score = 0.0

    # ── SIGNAL 3: RSI Recovery ──────────────────
    rsi_min_10d = rsi_series.rolling(10).min().iloc[-1]
    rsi_5d_ago = rsi_series.iloc[-6] if len(rsi_series) > 5 else rsi
    rsi_slope = (rsi - rsi_5d_ago) / 5

    if rsi_min_10d < 45 and rsi > 55:
        rsi_recovery_triggered = True
        s3_score = min((rsi-55)/3, 5) + min(rsi_slope*8, 5)
        s3_score = max(s3_score, 0)
    elif rsi_min_10d < 50 and rsi > 58 and rsi_slope > 1:
        rsi_recovery_triggered = True
        s3_score = min((rsi-55)/4, 3) + min(rsi_slope*5, 3)
        s3_score = max(s3_score, 0)
    else:
        rsi_recovery_triggered = False
        s3_score = 0.0

    # ── SIGNAL 4: Price Momentum ────────────────
    price_5d_ago = float(df["close"].iloc[-6]) if len(df) > 5 else close
    price_change_5d = ((close - price_5d_ago) / price_5d_ago) * 100
    price_1d_change = ((close - float(df["close"].iloc[-2]))
                       / float(df["close"].iloc[-2])) * 100

    if price_change_5d > 2.0 and price_1d_change > 0.3:
        momentum_triggered = True
        s4_score = min(price_change_5d * 0.8, 5)
    elif price_change_5d > 1.0:
        momentum_triggered = True
        s4_score = min(price_change_5d * 0.5, 3)
    else:
        momentum_triggered = False
        s4_score = 0.0

    # ── CONFLICT FLAG ───────────────────────────
    rsi_overbought = rsi > 70
    conflict_flag = breakout_triggered and rsi_overbought

    # ── COMBINED SCORE ──────────────────────────
    s1 = s1_score * 0.35 if breakout_triggered else 0.0
    s2 = s2_score * 0.25 if volume_spike_triggered else 0.0
    s3 = s3_score * 0.25 if rsi_recovery_triggered else 0.0
    s4 = s4_score * 0.15 if momentum_triggered else 0.0
    combined_score = round(s1 + s2 + s3 + s4, 2)

    # ── AGENT REASONING ─────────────────────────
    reasons = []
    if breakout_triggered:
        reasons.append(f"Broke {breakout_level} "
                       f"(+{breakout_pct:.1f}%)")
    if volume_spike_triggered:
        reasons.append(f"Volume {volume_ratio:.1f}x avg")
    if rsi_recovery_triggered:
        reasons.append(f"RSI recovering ({rsi:.0f})")
    if momentum_triggered:
        reasons.append(f"5d momentum +{price_change_5d:.1f}%")
    if conflict_flag:
        reasons.append("Caution: RSI overbought")

    agent_reasoning = (
        " · ".join(reasons) if reasons
        else "No strong setup — monitoring only"
    )

    is_fallback = combined_score < 1.0

    return {
        "symbol": symbol.replace(".NS", ""),
        "date": str(df.index[-1].date()),
        "breakout": bool(breakout_triggered),
        "volume_spike": bool(volume_spike_triggered),
        "rsi_recovery": bool(rsi_recovery_triggered),
        "momentum": bool(momentum_triggered),
        "conflict_flag": bool(conflict_flag),
        "s1_score": round(s1_score, 2),
        "s2_score": round(s2_score, 2),
        "s3_score": round(s3_score, 2),
        "s4_score": round(s4_score, 2),
        "combined_score": combined_score,
        "close": round(close, 2),
        "resistance_level": round(
            rolling_20_high if not pd.isna(rolling_20_high)
            else close * 0.98, 2),
        "volume_ratio": round(volume_ratio, 2),
        "rsi_today": round(rsi, 2),
        "price_change_5d": round(price_change_5d, 2),
        "price_1d_change": round(price_1d_change, 2),
        "agent_reasoning": agent_reasoning,
        "is_fallback": is_fallback,
        "video_path": "",
        "breakout_level": breakout_level,
    }

def build_fallback_signals(universe: dict, existing_results: list, min_count: int = 3) -> list:
    """
    Computes lightweight fallback metrics for stocks displaying early momentum 
    if algorithmic filters generate too few final actionable records.
    """
    existing_symbols = {r.get("symbol") for r in existing_results}
    candidates = []
    
    for symbol, df in universe.items():
        sym_clean = symbol.replace(".NS", "")
        if sym_clean in existing_symbols:
            continue
            
        if df is None or df.empty:
            continue
            
        try:
            last_row = df.iloc[-1]
            if len(df) > 1:
                prev_row = df.iloc[-2]
                prev_close = float(prev_row.get("close", 1.0))
            else:
                prev_close = float(last_row.get("open", 1.0))
                
            today_close = float(last_row.get("close", 1.0))
            price_momentum_pct = ((today_close - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
                
            volume_ratio = float(last_row.get("volume_ratio", 1.0))
            green_candle = int(last_row.get("green_candle", 0))
            rsi = float(last_row.get("rsi", 50.0))
            rolling_20_high = float(last_row.get("rolling_20_high", today_close))
            
            base = 3.0
            bonus_gc = 0.8 if green_candle == 1 else 0.0
            bonus_vr = min(max((volume_ratio - 1.0), 0.0), 1.0) * 0.7
            bonus_pm = min(max(price_momentum_pct, 0.0), 2.0) * 0.4
            
            fg_score = min(base + bonus_gc + bonus_vr + bonus_pm, 4.5)
            
            candidates.append({
                "symbol": sym_clean,
                "date": str(df.index[-1].date()),
                "breakout": False,
                "volume_spike": False,
                "rsi_recovery": False,
                "conflict_flag": False,
                "s1_score": 0.0,
                "s2_score": 0.0,
                "s3_score": 0.0,
                "combined_score": round(fg_score, 2),
                "close": round(today_close, 2),
                "resistance_level": round(rolling_20_high, 2),
                "volume_ratio": round(volume_ratio, 2),
                "rsi_today": round(rsi, 2),
                "agent_reasoning": "Early momentum detected, but no strong confirmed signal yet.",
                "is_fallback": True,
                "signals_triggered": ["Weak Momentum"],
                "price_momentum_pct": price_momentum_pct
            })
        except Exception:
            continue
            
    # Priority sorting internally by pure numerical momentum vectors
    candidates.sort(key=lambda x: (x.get("volume_ratio", 0), x.get("price_momentum_pct", 0)), reverse=True)
    
    needed = min_count - len(existing_results)
    if needed <= 0:
        return []
        
    final_fallbacks = candidates[:needed]
    for f in final_fallbacks:
        f.pop("price_momentum_pct", None)
        
    return final_fallbacks

def detect_all(universe: dict) -> list:
    """
    Applies detect_signals sequentially across the dictionary universe.
    
    Args:
        universe (dict): Mapping of tickers to their ready dataframes.
        
    Returns:
        list[dict]: Curated and sorted list of generated signal configurations.
    """
    results = []
    for symbol, df in universe.items():
        signal = detect_signals(symbol, df)
        if signal is not None:
            results.append(signal)
            
    # Sort firmly descending using combined_score natively mapping real properties
    results = sorted(results, key=lambda x: x.get("combined_score", 0.0), reverse=True)
    
    # ----------------------------------------------------
    # NEW: Conditionally append dynamically generated fallback placeholders
    # ----------------------------------------------------
    if len(results) < 3:
        fallbacks = build_fallback_signals(universe, results, min_count=3)
        fallbacks = sorted(fallbacks, key=lambda x: x.get("combined_score", 0.0), reverse=True)
        results.extend(fallbacks)
        
    return results
