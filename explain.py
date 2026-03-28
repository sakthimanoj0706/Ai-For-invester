import os

COMPANY_MAP = {
    "RELIANCE": "Reliance",
    "TCS": "TCS",
    "HDFCBANK": "HDFC Bank",
    "INFY": "Infosys",
    "ICICIBANK": "ICICI Bank",
    "TATAMOTORS": "Tata Motors",
    "MARUTI": "Maruti Suzuki",
    "BAJFINANCE": "Bajaj Finance",
    "SUNPHARMA": "Sun Pharma",
    "DRREDDY": "Dr. Reddy's",
    "CIPLA": "Cipla",
    "ADANIENT": "Adani Enterprises",
    "NTPC": "NTPC",
    "POWERGRID": "Power Grid",
    "WIPRO": "Wipro"
}

import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def call_llm(prompt: str,
             system: str,
             mode: str = "standard") -> str:
    """
    Primary LLM caller using Groq (free, fast).
    Uses Llama 3.1 8B — generous free tier limits.
    Falls back to template if unavailable.

    Args:
        prompt: signal data for the model
        system: system instruction string
        mode: standard / balanced / distress
    Returns:
        str: generated explanation text
    """
    api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        print("[LLM] GROQ_API_KEY not in .env")
        print("[LLM] Get free key: https://console.groq.com")
        return _template_fallback(mode)

    for attempt in range(2):
        try:
            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.7,
            )

            result = (response.choices[0]
                      .message.content.strip())
            print(f"[LLM] Groq responded: "
                  f"{len(result)} chars")
            return result

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 8 * (attempt + 1)
                print(f"[LLM] Rate limit. "
                      f"Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"[LLM] Groq error: {e}")
                break

    print("[LLM] Using template fallback")
    return _template_fallback(mode)


def _template_fallback(mode: str) -> str:
    """
    Safe hardcoded fallback.
    No disclaimer here — added by generate_explanation().
    """
    if mode == "balanced":
        return (
            "Breakout signal detected but RSI is "
            "overbought above 70. Wait for volume "
            "confirmation before acting."
        )
    elif mode == "distress":
        return (
            "Promoter bulk deal at discount detected. "
            "Possible distress selling. "
            "Monitor closely before acting."
        )
    else:
        return (
            "Early momentum signals detected. "
            "Watch for volume confirmation "
            "in next sessions."
        )


def generate_explanation(signal: dict,
                         win_rates: dict = None,
                         language: str = "en") -> str:
    
    if win_rates is None: win_rates = {}
    
    mode = signal.get("mode", "standard")
    symbol = signal.get("symbol", "")
    score = signal.get("combined_score", 0.0)
    close = signal.get("close", 0.0)
    vol_ratio = signal.get("volume_ratio", 0.0)
    rsi = signal.get("rsi_today", 0.0)
    resistance = signal.get("resistance_level", 0.0)
    signals_triggered = signal.get("signals_triggered", [])
    
    # Get win rates safely
    breakout_wr = win_rates.get("breakout", {}).get("win_rate", 0.68)
    volume_wr = win_rates.get("volume_spike", {}).get("win_rate", 0.61)
    rsi_wr = win_rates.get("rsi_recovery", {}).get("win_rate", 0.72)
    
    # Build user prompt with all signal data
    user_prompt = f"""
Stock: {symbol}
Current Price: ₹{close}
Signal Score: {score}/10
Signals Triggered: {', '.join(signals_triggered) if signals_triggered else 'None'}
Volume Ratio: {vol_ratio:.2f}x average
RSI Today: {rsi:.1f}
Resistance Level: ₹{resistance}
Breakout Win Rate (historical): {breakout_wr*100:.0f}%
Volume Spike Win Rate: {volume_wr*100:.0f}%
RSI Recovery Win Rate: {rsi_wr*100:.0f}%
Conflict Flag: {signal.get('conflict_flag', False)}
Is Fallback: {signal.get('is_fallback', False)}
"""
    
    # System prompts per mode
    if mode == "balanced":
        system = """You are a careful Indian stock market analyst.
Write a 70-word video voiceover script.
Rules:
- Acknowledge the bullish signal first
- Then clearly state the conflicting indicator  
- Quote the back-tested win rate
- Give a balanced watch level — not buy or sell
- Plain English, no jargon, no hype
- End with compliance disclaimer"""

    elif mode == "distress":
        system = """You are a risk-aware Indian market analyst.
Write a 70-word alert script.
Rules:
- Name the company and bulk deal facts
- State whether distress or routine selling
- Recommend caution with specific monitor level
- Plain English only
- End with compliance disclaimer"""
        
    else:  # standard
        system = """You are a concise Indian stock market analyst.
Write a 60-word video voiceover script.
Rules:
- Start with company name and what happened today
- Mention the specific signal type
- Include one specific number (price or ratio)
- Include the back-tested win rate
- End with: Watch for [specific level].
- Plain English only, no jargon"""

    # Language instruction
    if language == "ta":
        system += "\nRespond entirely in Tamil. Keep financial terms in English."
    elif language == "hi":
        system += "\nRespond entirely in Hindi. Keep financial terms in English."
    
    # Always append compliance
    compliance = "\n⚠ AI-generated analysis. Not licensed investment advice. Consult SEBI-registered advisor."
    
    # Use the new call_llm() function
    result = call_llm(
        prompt=user_prompt,
        system=system,
        mode=mode
    )
    
    if compliance not in result:
        result = result + compliance
        
    return result


def generate_all_languages(signal: dict,
                           win_rates: dict) -> dict:
    """
    Generates explanation in English, Tamil, Hindi.
    Adds delay between calls to respect rate limits.
    """
    import time
    explanations = {}

    for i, lang in enumerate(["en", "ta", "hi"]):
        try:
            # Small delay between calls to avoid 429
            if i > 0:
                time.sleep(6)

            explanations[lang] = generate_explanation(
                signal, win_rates, language=lang
            )
            print(f"[LLM] Generated {lang} explanation")

        except Exception as e:
            print(f"[LLM] {lang} failed: {e}")
            explanations[lang] = explanations.get(
                "en", "Explanation unavailable."
            )

    return explanations
