"""
models/llm.py
─────────────
Factory functions for all supported LLM providers.
Each function returns a LangChain-compatible chat model instance.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY,
    DEFAULT_OPENAI_MODEL, DEFAULT_GROQ_MODEL, DEFAULT_GEMINI_MODEL,
)


def get_openai_model(model_name: str = DEFAULT_OPENAI_MODEL):
    """
    Return a LangChain ChatOpenAI instance.
    Requires: OPENAI_API_KEY
    """
    try:
        from langchain_openai import ChatOpenAI
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in config/config.py")
        return ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            streaming=True,
        )
    except Exception as e:
        raise RuntimeError(f"[LLM] Failed to initialise OpenAI model '{model_name}': {e}")


def get_chatgroq_model(model_name: str = DEFAULT_GROQ_MODEL):
    """
    Return a LangChain ChatGroq instance.
    Requires: GROQ_API_KEY
    """
    try:
        from langchain_groq import ChatGroq
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in config/config.py")
        return ChatGroq(
            model=model_name,
            api_key=GROQ_API_KEY,
            temperature=0.7,
            streaming=True,
        )
    except Exception as e:
        raise RuntimeError(f"[LLM] Failed to initialise Groq model '{model_name}': {e}")


def get_gemini_model(model_name: str = DEFAULT_GEMINI_MODEL):
    """
    Return a LangChain ChatGoogleGenerativeAI instance.
    Requires: GEMINI_API_KEY
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in config/config.py")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            temperature=0.7,
            streaming=True,
        )
    except Exception as e:
        raise RuntimeError(f"[LLM] Failed to initialise Gemini model '{model_name}': {e}")


def get_model(provider: str, model_name: str = None):
    """
    Unified entry point — returns a chat model for the given provider string.
    provider: 'openai' | 'groq' | 'gemini'
    """
    provider = provider.lower().strip()
    try:
        if provider == "openai":
            return get_openai_model(model_name or DEFAULT_OPENAI_MODEL)
        elif provider == "groq":
            return get_chatgroq_model(model_name or DEFAULT_GROQ_MODEL)
        elif provider == "gemini":
            return get_gemini_model(model_name or DEFAULT_GEMINI_MODEL)
        else:
            raise ValueError(f"Unknown provider '{provider}'. Choose: openai | groq | gemini")
    except Exception as e:
        raise RuntimeError(f"[LLM] get_model error: {e}")
