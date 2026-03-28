import os
from datetime import datetime, timedelta

def get_today() -> str:
    """
    Returns today's date formatted as a string.
    
    Returns:
        str: Date in "YYYY-MM-DD" format.
    """
    return datetime.today().strftime("%Y-%m-%d")

def get_date_n_days_ago(n: int) -> str:
    """
    Returns the date `n` trading days ago (approximated by skipping weekends).
    
    Args:
        n (int): Number of trading days to go back.
        
    Returns:
        str: Date in "YYYY-MM-DD" format.
    """
    date = datetime.today()
    days_to_subtract = n
    while days_to_subtract > 0:
        date -= timedelta(days=1)
        if date.weekday() < 5:  # 0 to 4 are Monday to Friday
            days_to_subtract -= 1
    return date.strftime("%Y-%m-%d")

def ensure_dirs() -> None:
    """
    Creates necessary project directories if they don't already exist.
    Will silently skip if they persist.
    """
    directories = [
        "data",
        "outputs/videos",
        "outputs/audio"
    ]
    for d in directories:
        try:
            os.makedirs(d, exist_ok=True)
            print(f"[utils] Checked directory: {d}")
        except Exception as e:
            print(f"[utils] Error creating directory {d}: {e}")

def format_score(score: float) -> str:
    """
    Formats the score out of 10 along with its confidence label.
    
    Args:
        score (float): Score between 0.0 and 10.0.
        
    Returns:
        str: Formatted score (e.g., "8.2 / 10 (High)").
    """
    if score < 3.0:
        confidence = "Low"
    elif score <= 6.0:  # Matches 3-6 rule from prompt
        confidence = "Medium"
    else:
        confidence = "High"
    return f"{score:.1f} / 10 ({confidence})"

def get_sector(symbol: str) -> str:
    """
    Maps a stock symbol to its corresponding market sector.
    
    Args:
        symbol (str): NSE stock ticker (e.g., "RELIANCE.NS").
        
    Returns:
        str: The sector name, or "Unknown" if not found.
    """
    sector_map = {
        "RELIANCE.NS": "Energy", "ADANIENT.NS": "Energy", "NTPC.NS": "Energy", "POWERGRID.NS": "Energy",
        "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT",
        "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "BAJFINANCE.NS": "Banking",
        "TATAMOTORS.NS": "Auto", "MARUTI.NS": "Auto",
        "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma"
    }
    return sector_map.get(symbol, "Unknown")

def ticker_to_name(symbol: str) -> str:
    """
    Maps a stock symbol to its full company name.
    
    Args:
        symbol (str): NSE stock ticker.
        
    Returns:
        str: Full company name.
    """
    name_map = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "HDFCBANK.NS": "HDFC Bank",
        "INFY.NS": "Infosys",
        "ICICIBANK.NS": "ICICI Bank",
        "TATAMOTORS.NS": "Tata Motors",
        "MARUTI.NS": "Maruti Suzuki",
        "BAJFINANCE.NS": "Bajaj Finance",
        "SUNPHARMA.NS": "Sun Pharmaceuticals",
        "DRREDDY.NS": "Dr. Reddy's Laboratories",
        "CIPLA.NS": "Cipla",
        "ADANIENT.NS": "Adani Enterprises",
        "NTPC.NS": "NTPC Limited",
        "POWERGRID.NS": "Power Grid Corporation",
        "WIPRO.NS": "Wipro"
    }
    return name_map.get(symbol, symbol)

def generate_llm_response(prompt: str) -> str:
    import requests
    from src.gemini_helper import generate_with_gemini
            
    # Step 5 - Universal prepend rule
    system_intro = "You are a financial assistant. Keep output short, factual, and clear. Do not give advice.\n\n"
    full_prompt = system_intro + prompt
    
    # 1. TRY OLLAMA (PRIMARY)
    print("[LLM] Using Ollama")
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3",
            "prompt": full_prompt,
            "stream": False
        }
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
    except Exception:
        pass
        
    # 2. IF OLLAMA FAILS -> TRY GEMINI
    print("[LLM] Falling back to Gemini")
    gemini_response = generate_with_gemini(full_prompt)
    if gemini_response:
        return gemini_response
        
    # 3. IF BOTH FAIL
    print("[LLM] Using template fallback")
    return None
