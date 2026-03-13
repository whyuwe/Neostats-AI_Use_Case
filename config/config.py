"""
config/config.py
────────────────
Central configuration for all API keys and application settings.
All secrets must be stored here as environment variables — never hard-coded.
"""

import os

# ── LLM Provider API Keys ─────────────────────────────────────────────────────
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")

# ── Web Search API Key (Tavily) ───────────────────────────────────────────────
TAVILY_API_KEY   = os.environ.get("TAVILY_API_KEY", "")

# ── Default Model Names ───────────────────────────────────────────────────────
DEFAULT_OPENAI_MODEL  = "gpt-4o-mini"
DEFAULT_GROQ_MODEL    = "llama3-70b-8192"
DEFAULT_GEMINI_MODEL  = "gemini-1.5-flash"

# ── Embedding Settings ────────────────────────────────────────────────────────
EMBEDDING_MODEL        = "all-MiniLM-L6-v2"   # sentence-transformers model (local, free)
CHUNK_SIZE             = 500                    # characters per text chunk
CHUNK_OVERLAP          = 50                     # overlap between chunks
TOP_K_RESULTS          = 4                      # number of chunks to retrieve

# ── Vector Store ─────────────────────────────────────────────────────────────
VECTORSTORE_DIR        = "data/vectorstore"     # persisted FAISS index directory

# ── Response Mode Prompts ─────────────────────────────────────────────────────
CONCISE_INSTRUCTION  = "Reply in 2-3 sentences maximum. Be direct and to the point."
DETAILED_INSTRUCTION = "Provide a thorough, well-structured response with examples where helpful."

# ── Application Settings ──────────────────────────────────────────────────────
APP_TITLE    = "NeoStats AI Chatbot"
APP_ICON     = "🤖"
MAX_HISTORY  = 20   # max conversation turns kept in memory
