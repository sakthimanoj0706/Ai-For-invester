import pandas as pd

def analyze_bulk_deals(df: pd.DataFrame, previous_close_map: dict = None) -> dict:
    """
    Evaluates raw historical bulk deal dataframe logs for institutional flows.
    
    Args:
        df (pd.DataFrame): Dataframe fetched externally (usually from ingest.py).
        previous_close_map (dict): Optional dictionary mapping symbols to previous 
                                   day closing prices for distress detection checking.
                                   
    Returns:
        dict: Processed mapping dictionary analyzing flows per stock.
    """
    if df is None or df.empty:
        return {}
        
    if previous_close_map is None:
        previous_close_map = {}
        
    # Standardize column headers to lower-case dynamically
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    # Guard clauses against unexpectedly parsed HTML tables returning wrong shapes
    req_cols = ["stock_name", "deal_type", "quantity", "price"]
    for c in req_cols:
        if c not in df.columns:
            return {}
            
    # Safely coerce columns
    df["stock_name"] = df["stock_name"].astype(str).str.upper().str.strip()
    df["stock_name"] = df["stock_name"].str.replace(".NS", "", regex=False)
    
    df["deal_type"] = df["deal_type"].astype(str).str.upper().str.strip()
    
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    
    results = {}
    
    for stock, group in df.groupby("stock_name"):
        # Catch unexpected NaN categories explicitly
        if not stock or stock.lower() == "nan":
            continue
            
        total_qty = float(group["quantity"].sum())
        
        # Division catch safely
        if total_qty <= 0:
            continue
            
        # Weighted average transaction price
        avg_price = float((group["price"] * group["quantity"]).sum() / total_qty)
        
        # Handle "BUY", "Buy", "B", and equivalent string representations securely
        buy_mask = group["deal_type"].str.startswith("B")
        sell_mask = group["deal_type"].str.startswith("S")
        
        buy_qty = float(group.loc[buy_mask, "quantity"].sum())
        sell_qty = float(group.loc[sell_mask, "quantity"].sum())
        
        net_flow = buy_qty - sell_qty
        
        # Flow Flags
        accumulation = bool(net_flow > 0 and (buy_qty > 1.5 * sell_qty))
        distribution = bool(net_flow < 0 and (sell_qty > 1.5 * buy_qty))
        
        # Distress Flag Implementation
        distress = False
        prev_close = previous_close_map.get(stock)
        
        if prev_close and prev_close > 0:
            if distribution and (avg_price < (prev_close * 0.97)):
                distress = True
                
        results[stock] = {
            "net_flow": net_flow,
            "buy_qty": buy_qty,
            "sell_qty": sell_qty,
            "accumulation": accumulation,
            "distribution": distribution,
            "distress_flag": distress
        }
        
    return results

def merge_bulk_signal(signal: dict, bulk_data: dict) -> dict:
    """
    Appends Institutional flow context explicitly into an active signal dictionary safely.
    
    Args:
        signal (dict): Reference signal dict (usually outputted by Signals.py).
        bulk_data (dict): Reference block dictionary mapping extracted from analyze_bulk_deals.
        
    Returns:
        dict: Augmented signal.
    """
    if not isinstance(signal, dict):
        return {}
        
    sig = dict(signal)
    symbol = sig.get("symbol", "").replace(".NS", "")
    
    if symbol in bulk_data:
        data = bulk_data[symbol]
        sig["accumulation_flag"] = data.get("accumulation", False)
        sig["distribution_flag"] = data.get("distribution", False)
        sig["distress_flag"] = data.get("distress_flag", False)
    else:
        sig["accumulation_flag"] = False
        sig["distribution_flag"] = False
        sig["distress_flag"] = False
        
    return sig
