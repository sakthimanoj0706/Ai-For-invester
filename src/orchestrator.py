from src.signals import detect_all
from src.scoring import rank_signals
from src.ingest import fetch_bulk_deals
from src.bulk_agent import analyze_bulk_deals, merge_bulk_signal
from src.news_agent import fetch_news, analyze_sentiment, generate_news_summary, merge_news

def apply_recommendation_logic(signal: dict) -> dict:
    """
    Evaluates algorithmic footprints directly translating conflicts, distress,
    accumulation, and textual AI sentiments into a final composite action statement.
    
    Args:
        signal (dict): Merged payload mapped from downstream signal processors.
        
    Returns:
        dict: Processed signal encapsulating the string standard "recommendation_status".
    """
    if not isinstance(signal, dict):
        return {}
        
    sig = dict(signal)
    
    # Expose state metrics cleanly
    conflict_flag = sig.get("conflict_flag", False)
    distress_flag = sig.get("distress_flag", False)
    accumulation_flag = sig.get("accumulation_flag", False)
    distribution_flag = sig.get("distribution_flag", False)
    news_sentiment = str(sig.get("news_sentiment", "Neutral"))
    
    # Priority Resolution Flow
    if conflict_flag:
        status = "Caution"
    elif distress_flag:
        status = "High Risk"
    elif accumulation_flag and news_sentiment == "Positive":
        status = "Strong Watch"
    elif distribution_flag and news_sentiment == "Negative":
        status = "Avoid Watch"
    else:
        status = "Normal Watch"
        
    sig["recommendation_status"] = status
    return sig

def sort_final_signals(signals: list) -> list:
    """
    Pipelines the dataset logically filtering via scoring magnitudes seamlessly
    coupled statistically next to the categorical status prioritization.
    
    Args:
        signals (list): Fully saturated universe pool mappings.
        
    Returns:
        list: Highest rigorously graded payload subset cleanly arrayed to top 3 outputs.
    """
    if not isinstance(signals, list) or not signals:
        return []
        
    # Dictionary lookup matrix resolving category string logic statistically
    status_priority = {
        "Strong Watch": 5,
        "Normal Watch": 4,
        "Caution": 3,
        "High Risk": 2,
        "Avoid Watch": 1
    }
    
    def sorting_key(x):
        return (
            float(x.get("combined_score", 0.0)),
            status_priority.get(x.get("recommendation_status", "Normal Watch"), 0)
        )
        
    # Python arrays evaluate tuple sorts hierarchically precisely natively
    sorted_sigs = sorted(signals, key=sorting_key, reverse=True)
    return sorted_sigs[:3]

def run_pipeline(universe: dict) -> list:
    """
    The central conductor pipeline actively connecting isolated analytical units.
    Executes technical validations globally, subsequently orchestrating detailed 
    bulk and NLP news processing isolated tightly against ranked outputs alone to 
    safeguard LLM budgets optimally.
    
    Args:
        universe (dict): Dictionary mapping stock identifiers universally to localized DataFrames.
        
    Returns:
        list[dict]: Array bounding precisely 3 globally validated operational intelligence outputs.
    """
    # 1. Broad Technical Screening
    try:
        raw_signals = detect_all(universe)
    except Exception as e:
        print(f"[orchestrator] Critical engine failure identifying base indicators: {e}")
        return []

    # 2. Narrow Field Extrapolations (Limits NLP overhead intelligently)
    try:
        ranked_signals = rank_signals(raw_signals, top_n=10)
    except Exception as e:
        print(f"[orchestrator] Failure tracking sorting scopes: {e}")
        return []

    if not ranked_signals:
        return []

    # 3. Aggregate Macro Structure Context natively
    bulk_data = {}
    try:
        bulk_df = fetch_bulk_deals()
        bulk_data = analyze_bulk_deals(bulk_df)
    except Exception as e:
        print(f"[orchestrator] Notice: External Bulk processing bypassed harmlessly. {e}")
        
    finalized_signals = []
    
    # 4. Integrate parallel matrices sequentially onto subset exclusively
    for signal in ranked_signals:
        symbol = signal.get("symbol", "Unknown")
        try:
            # Safely clone and map Institutional Context explicitly
            sig = merge_bulk_signal(signal, bulk_data)
            
            # Setup resilient fallback properties naturally
            sentiment_dict = {"sentiment": "Neutral", "score": 0, "headline_count": 0}
            summary_statement = "No major fresh news catalyst detected today. Current setup is driven primarily by market behavior."
            
            # Attaching Media Context intelligently
            try:
                news_list = fetch_news(symbol)
                # Ensure the NLP analyzer fires strictly against valid HTTP arrays
                if news_list and isinstance(news_list, list) and len(news_list) > 0:
                    sentiment_dict = analyze_sentiment(news_list)
                    summary_statement = generate_news_summary(symbol, sentiment_dict, news_list)
            except Exception as news_error:
                print(f"[orchestrator] AI Enrichment degraded cleanly for {symbol}: {news_error}")
                
            sig = merge_news(sig, summary_statement, sentiment_dict)
            
            # 5. Composite Finalization
            sig = apply_recommendation_logic(sig)
            finalized_signals.append(sig)
            
        except Exception as iter_e:
            print(f"[orchestrator] Loop sequence iteration crashed safely skipping {symbol}: {iter_e}")
            continue
            
    # 6 & 7. Output Routing Priority Matrix Output explicitly
    sorted_sigs = sort_final_signals(finalized_signals)
    if sorted_sigs:
        real_sigs = [s for s in sorted_sigs if not s.get("is_fallback", False)]
        if real_sigs:
            top_sig = max(real_sigs, key=lambda x: x.get("combined_score", 0))
        else:
            top_sig = sorted_sigs[0]
        top_sig["is_top_opportunity"] = True
    return sorted_sigs
