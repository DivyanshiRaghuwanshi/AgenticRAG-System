import os
from dotenv import load_dotenv

# Use override=True so it strictly uses your .env file and ignores any old hidden Windows system variables
load_dotenv(override=True)

# API keys — read from Streamlit secrets first, then env vars, then empty string.
# For local dev: add keys to .streamlit/secrets.toml (already in .gitignore)
# For Streamlit Cloud: paste keys in App Settings → Secrets
def _secret(key):
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, ""))
    except Exception:
        return os.environ.get(key, "")

# GROQ_API_KEY   = _secret("GROQ_API_KEY")
# OPENAI_API_KEY = _secret("OPENAI_API_KEY")
# GEMINI_API_KEY = _secret("GEMINI_API_KEY")
# SERPER_API_KEY = _secret("SERPER_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# LLM settings
DEFAULT_LLM_PROVIDER = "gemini"

GROQ_MODEL_NAME   = "llama-3.1-8b-instant"
OPENAI_MODEL_NAME = "gpt-4o-mini"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
OPENROUTER_MODEL_NAME = "openai/gpt-5.4-pro"
OLLAMA_MODEL_NAME = "llama3.2"  # Local self-hosted fallback model

# temperature=0 for retrieval (factual), 0.3 for response (slight creativity)
RETRIEVAL_TEMPERATURE = 0.0
RESPONSE_TEMPERATURE  = 0.3

# RAG settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K         = 7   # KNN top-k chunks to retrieve

# Keep last N messages in context to avoid hitting token limits
MAX_HISTORY_MESSAGES = 20

SERPER_NUM_RESULTS = 5

