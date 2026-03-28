import os
import requests
import xml.etree.ElementTree as ET
from src.utils import ticker_to_name

def fetch_news(symbol: str) -> list:
    """
    Scrapes or fetches cleanly up to 5 latest articles regarding a stock 
    via NewsAPI or Google News RSS as a silent fallback.
    
    Args:
        symbol (str): Target stock ticker.
        
    Returns:
        list[dict]: Uncrashed JSON array of articles.
    """
    company = ticker_to_name(symbol)
    query = f"{company} stock India"
    results = []
    
    # Optional NewsAPI Attempt
    try:
        api_key = os.environ.get("NEWS_API_KEY")
        if api_key:
            url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={api_key}"
            res = requests.get(url, timeout=5)
            res.raise_for_status()
            
            articles = res.json().get("articles", [])[:5]
            for a in articles:
                results.append({
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "date": a.get("publishedAt", ""),
                    "url": a.get("url", "")
                })
            return results
    except Exception:
        pass
        
    # Standard RSS Fallback
    try:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        
        root = ET.fromstring(res.content)
        items = root.findall(".//item")[:5]
        
        for item in items:
            title = item.findtext("title", "")
            source = item.findtext("source", "")
            date = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            
            results.append({
                "title": title,
                "source": source,
                "date": date,
                "url": link
            })
            
        return results
    except Exception as e:
        print(f"[news_agent] Silent fetch failure on {symbol}: {e}")
        return []

def analyze_sentiment(news_list: list) -> dict:
    """
    Lightweight exact-match boolean scoring mechanism to establish momentum metrics.
    
    Args:
        news_list (list): The array block pulled previously.
        
    Returns:
        dict: Compiled sentiment configuration.
    """
    pos_keywords = ["growth", "profit", "expansion", "bullish", "strong", "upgrade"]
    neg_keywords = ["loss", "decline", "fraud", "fall", "bearish", "downgrade"]
    
    pos_count = 0
    neg_count = 0
    
    for item in news_list:
        title = str(item.get("title", "")).lower()
        
        for pk in pos_keywords:
            if pk in title:
                pos_count += 1
                
        for nk in neg_keywords:
            if nk in title:
                neg_count += 1
                
    net_score = pos_count - neg_count
    
    if net_score > 0:
        sentiment_label = "Positive"
    elif net_score < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
        
    return {
        "sentiment": sentiment_label,
        "score": net_score,
        "headline_count": len(news_list)
    }

def generate_news_summary(symbol: str, sentiment: dict, news_list: list) -> str:
    """
    Utilizes local arrays to compose a short natural NLP summation of stock action 
    directly constrained by headlines and keyword momentum.
    """
    fallback_text = "No major fresh news catalyst detected today. Current setup is driven primarily by market behavior."
    
    if not news_list:
        return fallback_text
        
    try:
        from src.utils import generate_llm_response
        company = ticker_to_name(symbol)
        
        headlines_text = ""
        for item in news_list[:2]:
            headlines_text += f"- {item.get('title', '')}\n"
            
        user_prompt = f"""
Include these details natively under 2 sentences max strictly formatted:
- stock name: {company} ({symbol})
- sentiment: {sentiment.get('sentiment', 'Neutral')}
- top 2 headlines:
{headlines_text}
"""
        response = generate_llm_response(user_prompt)
        if response:
             return str(response).strip()
             
    except Exception as e:
        print(f"[news_agent] Analytics LLM timeout/fault for {symbol}: {e}")
        
    return fallback_text

def merge_news(signal: dict, news_summary: str, sentiment: dict) -> dict:
    """
    Appends AI news summarizations perfectly onto active signal templates.
    """
    if not isinstance(signal, dict):
        return {}
        
    sig = dict(signal)
    sig["news_summary"] = news_summary
    sig["news_sentiment"] = sentiment.get("sentiment", "Neutral")
    
    return sig
