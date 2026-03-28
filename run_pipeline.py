from dotenv import load_dotenv
load_dotenv()

from src.ingest import fetch_universe
from src.indicators import add_indicators
from src.orchestrator import run_pipeline as orchestrate
from src.explain import generate_all_languages
from src.video_engine import generate_video
from src.db import init_db, insert_signal
from src.utils import get_today, ensure_dirs
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="google")

def run_pipeline(date_str: str = None, user_input_symbols: list = None) -> list:
    """
    Executes the comprehensive SignalAI macro-architecture from source extraction 
    down through statistical analysis, generative multiregional language translations, 
    graphical rendering, and database persistence sequentially without falter.
    
    Args:
        date_str (str): Overrides the master timestamp natively (YYYY-MM-DD). Defaults to today.
        
    Returns:
        list[dict]: Uncrashed array of maximally enriched dictionaries finalized post-compilation.
    """
    # 1. Pipeline Environment Setup
    ensure_dirs()
    init_db()
    if date_str is None:
        date_str = get_today()
        
    # 2. Data Extrusion & Math Layer
    print("[pipeline] Loading base asset universe...")
    universe = fetch_universe()

    enriched_universe = {}
    for symbol, df in universe.items():
        if df is not None and not df.empty:
            try:
                enriched_universe[symbol] = add_indicators(df)
            except Exception as e:
                print(f"[pipeline] Skipping mathematical evaluation for {symbol}: {e}")

    # 3. Macro Intelligence Coordination
    print("[pipeline] Initiating Central AI Orchestrator...")
    try:
        final_signals = orchestrate(enriched_universe)
    except Exception as e:
        print(f"[pipeline] Catastrophic failure traversing orchestrator: {e}")
        return []

    # 4. Action Iterators
    print(f"[pipeline] Processing {len(final_signals)} viable triggers through generative engines...")
    
    user_input_symbols = user_input_symbols or []

    # Identify top scoring stock strictly mapped by pipeline architecture
    if final_signals:
        try:
            top_symbol = max(final_signals, key=lambda x: x.get("score", x.get("combined_score", 0)))["symbol"]
        except Exception:
            top_symbol = final_signals[0]["symbol"]

        for s in final_signals:
            s["is_top_opportunity"] = (s["symbol"] == top_symbol)
            s["is_user_stock"] = (s["symbol"] in user_input_symbols)

    for i, signal in enumerate(final_signals):
        
        # Delay between signals to respect Gemini
        # rate limits (free tier: 15 req/min)
        if i > 0:
            import time
            print(f"[pipeline] Waiting 8s before next signal...")
            time.sleep(8)
        
        sym = signal.get("symbol", "Unknown")
        print(f"[pipeline] Processing {sym}...")
        
        # a & b. AI Semantic Transcriptions
        try:
            explanations = generate_all_languages(signal)
            signal["explanation_en"] = explanations.get("en", "")
            signal["explanation_ta"] = explanations.get("ta", "")
            signal["explanation_hi"] = explanations.get("hi", "")
        except Exception as exp_e:
            print(f"[pipeline] Text generation collapsed safely for {sym}: {exp_e}")
            signal["explanation_en"] = ""
            signal["explanation_ta"] = ""
            signal["explanation_hi"] = ""

        # STEP 1: Generate video first
        video_path = ""
        try:
            df_key = f"{sym}.NS"
            df = enriched_universe.get(df_key)
            if df is not None and not df.empty:
                video_path = generate_video(
                    signal=signal,
                    df=df,
                    explanation_text=signal.get(
                        "explanation_en", "")
                ) or ""
        except Exception as vid_e:
            print(f"[pipeline] Video error for {sym}: {vid_e}")
            video_path = ""
    
        # STEP 2: Verify file exists on disk
        import os
        if video_path and os.path.exists(str(video_path)):
            print(f"[pipeline] Video confirmed: {video_path}")
            signal["video_path"] = str(video_path)
        else:
            print(f"[pipeline] No video file for {sym}")
            signal["video_path"] = ""
    
        # STEP 3: Build DB record AFTER video_path is set
        try:
            sgnls = signal.get("signals_triggered", [])
            
            # If empty, build from individual boolean flags
            if not sgnls:
                sgnls = []
                if signal.get("breakout"):
                    sgnls.append("Breakout")
                if signal.get("volume_spike"):
                    sgnls.append("Volume Spike")
                if signal.get("rsi_recovery"):
                    sgnls.append("RSI Recovery")
            
            # Final fallback label
            if not sgnls:
                sgnls = ["Weak Momentum"]
            
            signals_str = ", ".join(sgnls)
            record = {
                "timestamp": str(date_str),
                "symbol": sym,
                "score": float(signal.get("combined_score", 0.0)),
                "confidence": str(signal.get("confidence", "")),
                "signals": signals_str,
                "conflict_flag": int(signal.get("conflict_flag", False)),
                "distress_flag": int(signal.get("distress_flag", False)),
                "video_path": signal["video_path"],  # ← uses real path
                "explanation_en": str(signal.get("explanation_en", "")),
                "explanation_ta": str(signal.get("explanation_ta", "")),
                "explanation_hi": str(signal.get("explanation_hi", "")),
                "agent_reasoning": str(signal.get("agent_reasoning", ""))
            }
            insert_signal(record)
        except Exception as db_e:
            print(f"[pipeline] DB error for {sym}: {db_e}")

    # 5. Output Summary Formats
    print("\n=== SignalAI Pipeline Complete ===")
    print(f"Date: {date_str}")
    print(f"Signals generated: {len(final_signals)}")

    if len(final_signals) > 0:
        top = final_signals[0]
        t_sym = top.get("symbol", "Unk")
        t_sco = top.get("combined_score", 0.0)
        t_rec = top.get("recommendation_status", "None")
        print(f"Top signal: {t_sym} — Score {t_sco} — {t_rec}")

    # 6. Returns List cleanly
    return final_signals

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SignalAI Action Generation Master Trigger")
    parser.add_argument("--date", type=str, default=None, help="Explicitly force pipeline timestamp parameter")
    args = parser.parse_args()
    
    run_pipeline(args.date)
