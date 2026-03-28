# 📈 SignalAI Setup Guide
AI-Powered Market Intelligence for the Indian Investor

This guide outlines exactly how to configure, securely authenticate, and launch the end-to-end framework.

## 1. Create virtual environment
It is highly recommended to cleanly isolate dependencies.
```bash
python -m venv venv
```

## 2. Activate environment
Select the exact command corresponding to your operating system:

**Windows:**
```cmd
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

## 3. Install dependencies
Ensure the environment is active (you will see `(venv)` in your terminal line) before mapping the pip modules.
```bash
pip install -r requirements.txt
```

## 4. Setup environment variables
SignalAI relies passively on API tokens to generate AI narratives. The system is structurally designed to natively bypass these calls via hardcoded fallbacks if you don't supply keys.
1. Copy the `.env.example` mapping and strip the extension:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill the placeholder strings if LLMs are preferred over fallbacks.

## 5. Run pipeline
To command the master orchestrator to scrape prices, compute equations, summarize audio/media, and inject SQLite databases locally:
```bash
python run_pipeline.py
```
*(Wait a few minutes. Videos will dynamically compile inside `outputs/videos/`)*

## 6. Run dashboard
Once the pipeline has seeded the SQL ledger natively, launch the Streamlit frontend presentation layer:
```bash
streamlit run app.py
```
*(The dashboard natively hosts securely at `http://localhost:8501`)*

---

## 7. Common fixes
- **`moviepy` rendering issues (Missing files/Errors):** A common dependency collision. You physically need the backend renderer installed via executing `pip install imageio-ffmpeg` to construct MP4 components properly.
- **`pandas-ta` missing or fault loop:** If indicators refuse to trace, the pip sequence likely skipped it due to a conflict. Override cleanly via `pip install --force-reinstall pandas-ta`.
- **`gTTS` Audio missing dynamically:** You must have an active internet connection running on the host machine to synthesize spoken phrases natively. If network drops, SignalAI bypasses the voiceover engine silently mapping a silent visual MP4 instead.
