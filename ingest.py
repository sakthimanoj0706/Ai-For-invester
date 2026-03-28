import os
import pandas as pd
from typing import Dict
from src.utils import ensure_dirs

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "TATAMOTORS.NS", "MARUTI.NS", "BAJFINANCE.NS", "SUNPHARMA.NS",
    "DRREDDY.NS", "CIPLA.NS", "ADANIENT.NS", "NTPC.NS", "POWERGRID.NS",
    "WIPRO.NS"
]

def fetch_stock(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Downloads OHLCV data via yfinance for a single stock.
    Falls back to local CSV cache on failure.
    
    Args:
        symbol (str): The ticker symbol to fetch.
        period (str): The time period string for yfinance.
        
    Returns:
        pd.DataFrame: Processed OHLCV data, or empty DataFrame if all attempts fail.
    """
    ensure_dirs()
    csv_path = os.path.join("data", f"{symbol}.csv")
    
    try:
        import yfinance as yf
        # Use Ticker.history to cleanly fetch single stock OHLCV 
        # avoiding newest yfinance MultiIndex download formats
        df = yf.Ticker(symbol).history(period=period)
        
        if df is None or df.empty:
            raise ValueError("Empty data returned by yfinance")
            
        # Rename to lowercase
        rename_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        }
        df = df.rename(columns=rename_map)
        
        # Keep strictly required columns
        valid_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[valid_cols]
        
        # Set datetime index, drop NaNs
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna()
        
        if df.empty:
            raise ValueError("All rows dropped due to NaNs")
            
        # Write through to cache
        df.to_csv(csv_path)
        print(f"Fetched {symbol}: {len(df)} rows")
        return df

    except Exception as e:
        # Fallback 1: cache
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
                print(f"Loaded {symbol} from cache")
                return df
            except Exception:
                pass
                
        # Fallback 2: Nothing
        print(f"Skipped {symbol}: no data")
        return pd.DataFrame()

def fetch_universe() -> Dict[str, pd.DataFrame]:
    """
    Calls fetch_stock for all 15 master tickers.
    
    Returns:
        Dict[str, pd.DataFrame]: Mapping of ticker symbols to their OHLCV DataFrames.
    """
    results = {}
    for symbol in TICKERS:
        df = fetch_stock(symbol)
        if not df.empty:
            results[symbol] = df
    return results

def fetch_bulk_deals() -> pd.DataFrame:
    """
    Attempts to download BSE bulk deal CSV.
    Safely fallback to an empty DataFrame with valid columns if it fails.
    
    Returns:
        pd.DataFrame: BSE Bulk deals data structure.
    """
    ensure_dirs()
    csv_path = os.path.join("data", "bulk_deals.csv")
    columns = [
        "date", "stock_name", "client_name", 
        "deal_type", "quantity", "price", "remarks"
    ]
    
    try:
        import requests
        import io
        url = "https://www.bseindia.com/markets/MarketInfo/BulkDealArchieve.aspx"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0",
            "Accept": "text/html,application/xhtml+xml,application/xml"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # In a real environment BSE relies on complex ASP.NET POST payloads for this route,
        # but we gracefully attempt pd.read_html first, falling back onto read_csv.
        try:
            tables = pd.read_html(io.StringIO(response.text))
            df = tables[0] if tables else pd.DataFrame(columns=columns)
            if len(df.columns) >= 7:
                df = df.iloc[:, :7]
                df.columns = columns
            else:
                df = pd.DataFrame(columns=columns)
        except Exception:
            try:
                df = pd.read_csv(io.StringIO(response.text))
                if len(df.columns) >= 7:
                    df = df.iloc[:, :7]
                    df.columns = columns
                else:
                    df = pd.DataFrame(columns=columns)
            except Exception:
                df = pd.DataFrame(columns=columns)
                
        # Must drop NA rows cleanly if site returns a generic 1-row junk text response
        df = df.dropna(how='all')
        
        if df.empty:
            raise ValueError("No valid bulk deal data found")
            
        df.to_csv(csv_path, index=False)
        return df
        
    except Exception as e:
        # Fallback to empty formatted dataframe
        return pd.DataFrame(columns=columns)
