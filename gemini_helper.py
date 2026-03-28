import os

def get_gemini_client():
    """
    Reads GEMINI_API_KEY from environment safely.
    Handles imports for both legacy generativeai and the new google.genai package styles.
    Returns a dictionary mapping the mode and the active client or None.
    Never crashes.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
            
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return {"mode": "legacy", "client": genai}
        except ImportError:
            try:
                import google.genai as genai
                client = genai.Client(api_key=api_key)
                return {"mode": "new", "client": client}
            except ImportError:
                return None
    except Exception as e:
        print(f"[gemini_helper] Initialization gracefully failed: {e}")
        return None

def generate_with_gemini(prompt: str) -> str | None:
    """
    Generates a short, factual text completion using gemini-1.5-flash.
    Safely bypasses all crashes returning None on exceptions.
    """
    try:
        helper = get_gemini_client()
        if not helper:
            return None
            
        if helper["mode"] == "legacy":
            model = helper["client"].GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        elif helper["mode"] == "new":
            response = helper["client"].models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            return response.text.strip()
    except Exception as e:
        print(f"[gemini_helper] Generation generation skipped due to: {e}")
        return None
        
def gemini_available() -> bool:
    """
    Checks if Gemini import mechanisms and API keys exist safely.
    Returns: bool
    """
    try:
        return get_gemini_client() is not None
    except Exception:
        return False
