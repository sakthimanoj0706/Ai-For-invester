import json
import os
import pandas as pd

from src.ingest import fetch_universe
from src.indicators import add_indicators
from src.signals import detect_signals
from src.db import insert_backtest, get_win_rates

def run_backtest(days: int = 365) -> None:
    """
    Simulates trading triggers strictly enforcing backwards data visibility,
    cataloging historical profitability via SQL insertions.
    
    Args:
        days (int): Depth limit backward in time to evaluate metrics.
    """
    print("[backtest] Commencing simulation sequence...")
    
    # 1. Fetch cleanly isolated full universe
    universe_df_dict = fetch_universe()

    for symbol, df in universe_df_dict.items():
        try:
            # 2. Vectorized indicator mapping to bypass massive loop recalculations
            #    Calculations only look functionally backward per rules.
            df = add_indicators(df)
            
            signals_generated_count = 0
            
            # Guard clause against heavily truncated historical inputs
            if len(df) <= 40:
                print(f"[backtest] Skipping {symbol.replace('.NS', '')}: insufficient data.")
                continue

            # 3. Configure bounds securely
            #    Reserve 30 start days for robust moving average seeding.
            #    Reserve 10 terminal days for future_close profitability measuring.
            loop_end = len(df) - 10
            
            # Optionally constrain lookback duration based on the 'days' threshold
            loop_start = max(30, loop_end - days)

            # 4. Step historical iterators 
            for i in range(loop_start, loop_end):
                try:
                    # Provide exclusively the historically constrained subset dataframe to signal logic
                    sub_df = df.iloc[:i+1]
                    
                    signal = detect_signals(symbol, sub_df)
                    
                    if signal is None:
                        continue
                        
                    # Calculate metric performance securely
                    signal_close = float(df["close"].iloc[i])
                    future_close = float(df["close"].iloc[i+10])
                    
                    return_10d = ((future_close - signal_close) / signal_close) * 100.0
                    was_profitable = 1 if return_10d > 0.0 else 0
                    
                    # Transform boolean triggers directly to comma strings for SQL storage
                    triggered_arr = []
                    if signal.get("breakout"): triggered_arr.append("breakout")
                    if signal.get("volume_spike"): triggered_arr.append("volume_spike")
                    if signal.get("rsi_recovery"): triggered_arr.append("rsi_recovery")
                    
                    sigs_str = ", ".join(triggered_arr)
                    if not sigs_str:
                        continue
                        
                    # 5. DB Persistence Mapping
                    record = {
                        "ticker": symbol.replace(".NS", ""),
                        "signal_date": str(df.index[i].date()),
                        "signals_triggered": sigs_str,
                        "signal_close": round(signal_close, 2),
                        "future_close": round(future_close, 2),
                        "return_10d": round(return_10d, 2),
                        "was_profitable": was_profitable
                    }
                    
                    insert_backtest(record)
                    signals_generated_count += 1
                    
                except Exception:
                    # Catch and resume loop to safely fulfill "Never crash on one stock failure"
                    continue
                    
            # 6. Structured terminal output format
            print(f"Backtesting {symbol.replace('.NS', '')}: {signals_generated_count} signals found")
            
        except Exception as e:
            print(f"[backtest] Aborted {symbol} due to catastrophic exception: {e}")

def compute_summary() -> dict:
    """
    Returns the compiled statistical evaluations utilizing the database win rates mapping.
    
    Returns:
        dict: Standard structured metrics matrix.
    """
    return get_win_rates()

def save_summary(path: str = "backtest_summary.json") -> None:
    """
    Downloads database aggregates natively dumping to standalone filesystem JSON format.
    
    Args:
        path (str): File destination formatting.
    """
    try:
        summary_payload = compute_summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=4)
        print(f"[backtest] Successfully compiled metrics directly to {path}")
    except Exception as e:
        print(f"[backtest] Critical failure outputting JSON: {e}")

if __name__ == "__main__":
    from src.db import init_db
    
    # Boilerplate execution loop
    init_db()
    run_backtest()
    save_summary()
    
    print("=== Backtest Complete ===")
