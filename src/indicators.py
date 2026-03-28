import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Wilder's Relative Strength Index (RSI) optimally utilizing Pandas vectors,
    eliminating restrictive external TA dependencies entirely.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Exponential moving metrics mapping Wilder natively
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Mathematically lock RSI limits gracefully bypassing nullities
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi.clip(0, 100)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates advanced technical indicators required for the signal engine.
    
    Args:
        df (pd.DataFrame): The base OHLCV structure.
        
    Returns:
        pd.DataFrame: Appended dataframe containing all 8 target indicator columns.
    """
    if df.empty:
        return df

    # Work on a copy to avoid mutation side-effects and SettingWithCopyWarnings
    df = df.copy()

    # 1. 20-day Rolling High (excluding today)
    df["rolling_20_high"] = df["high"].rolling(20).max().shift(1)

    # 2. 20-day Average Volume (excluding today)
    df["avg_volume_20"] = df["volume"].rolling(20).mean().shift(1)

    # 3. Volume Ratio (safely avoiding divide-by-zero by swapping 0 avg to NaN)
    df["volume_ratio"] = df["volume"] / df["avg_volume_20"].replace(0, np.nan)

    # 4. Green candle flag
    df["green_candle"] = (df["close"] > df["open"]).astype(int)

    # 5. RSI 14 (Native mathematical evaluation bypassing pandas-ta)
    df["rsi"] = compute_rsi(df["close"], period=14)

    # 6. RSI Overbought flag (1 if > 70, naturally evaluates NaN as False = 0)
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)

    # 7. RSI Was Oversold check (True if any day in last 10 went below 40)
    df["rsi_was_oversold"] = df["rsi"].rolling(10).min() < 40

    # 8. RSI Slope (5-day slope)
    df["rsi_slope"] = (df["rsi"] - df["rsi"].shift(5)) / 5

    return df
