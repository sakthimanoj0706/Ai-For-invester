def compute_score(signal: dict) -> dict:
    """
    Enriches the raw signal dictionary with human-readable confidence metrics and labels.
    
    Args:
        signal (dict): Raw signal generated from signals.py.
        
    Returns:
        dict: A new dictionary safely containing the appended fields.
    """
    if not isinstance(signal, dict):
        return {}
        
    sig = dict(signal)
    
    combined_score = sig.get("combined_score", 0.0)

    if combined_score < 2.0:
        confidence = "Low"
    elif combined_score < 3.5:
        confidence = "Watch"
    elif combined_score < 6.0:
        confidence = "Medium"
    else:
        confidence = "High"

    triggered = []
    if sig.get("breakout"):
        triggered.append("Breakout")
    if sig.get("volume_spike"):
        triggered.append("Volume Spike")
    if sig.get("rsi_recovery"):
        triggered.append("RSI Recovery")
    if sig.get("momentum"):
        triggered.append("Price Momentum")
    if not triggered:
        triggered = ["Weak Momentum"]
    sig["signals_triggered"] = triggered
    
    sig["confidence"] = confidence
    sig["score_label"] = f"{combined_score} / 10 ({confidence})"
    
    return sig

def get_watch_level(signal: dict) -> str:
    """
    Evaluates the signal footprint to output a precise algorithmic watch statement.
    Checks flags sequentially through standard priority flow.
    """
    conflict = signal.get("conflict_flag", False)
    breakout = signal.get("breakout", False)
    volume_spike = signal.get("volume_spike", False)
    rsi_recovery = signal.get("rsi_recovery", False)
    res_level = signal.get("resistance_level", 0.0)

    if conflict:
        return "Watch closely — conflicting signals present. Wait for confirmation."
    elif breakout:
        # Handle slightly malformed resistance values safely
        target = round(res_level * 1.02, 2) if res_level else 0.0
        return f"Watch for sustained move above {target}"
    elif volume_spike:
        return "Watch for volume to remain above 2x average in next sessions"
    elif rsi_recovery:
        return "Watch for RSI to sustain above 60 for continued momentum"
    else:
        return "No clear actionable level"

def rank_signals(signals: list, top_n: int = 3, holdings: list = None) -> list:
    """
    Pipelines the dataset: injects labels via compute_score, filters zeros, sorts,
    and crops to top N entries.
    
    Args:
        signals (list): List of standard dictionaries directly from detect_all.
        top_n (int): Defines maximum length of the return array.
        holdings (list): Optional list of holding symbols to prioritize.
        
    Returns:
        list: Highest-graded enriched signals.
    """
    if not signals:
        return []
        
    if holdings is None:
        holdings = []

    enriched = []
    for s in signals:
        try:
            # Enrich individual dict
            new_s = compute_score(s)
            
            # Filter condition: skip absolute zero scores entirely
            if new_s.get("combined_score", 0.0) > 0.0:
                enriched.append(new_s)
        except Exception as e:
            import logging
            logging.warning(f"SignalAI score error: {e}")
            
    # Sort strictly descending on score, prioritizing holdings
    enriched.sort(key=lambda x: (x.get("symbol") in holdings, x.get("combined_score", 0.0)), reverse=True)
    
    # Fallback: if nothing passed filter, inject best available
    if not enriched and signals:
        best = max(signals, key=lambda x: x.get("combined_score", 0.0))
        best_copy = compute_score(dict(best))
        best_copy["is_fallback"] = True
        enriched = [best_copy]
    
    # Slice to capacity
    return enriched[:top_n]
