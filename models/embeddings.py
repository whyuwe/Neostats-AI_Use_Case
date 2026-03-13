"""
models/embeddings.py
─────────────────────
Embedding model factory for RAG pipeline.
Uses sentence-transformers locally (no API key required).
Can be swapped for OpenAI embeddings when OPENAI_API_KEY is available.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import EMBEDDING_MODEL, OPENAI_API_KEY


def get_local_embeddings():
    """
    Returns a HuggingFace sentence-transformers embedding model.
    Runs locally — free, no API key required.
    Model: all-MiniLM-L6-v2 (fast, 384-dim, great for semantic search)
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(f"[Embeddings] Failed to load local HuggingFace embeddings: {e}")


def get_openai_embeddings():
    """
    Returns OpenAI text-embedding-ada-002 embeddings.
    Requires OPENAI_API_KEY.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    except Exception as e:
        raise RuntimeError(f"[Embeddings] Failed to load OpenAI embeddings: {e}")


def get_embeddings(use_openai: bool = False):
    """
    Unified entry point.
    use_openai=True  → OpenAI Ada embeddings (requires API key)
    use_openai=False → Local HuggingFace embeddings (default, free)
    """
    try:
        if use_openai and OPENAI_API_KEY:
            return get_openai_embeddings()
        return get_local_embeddings()
    except Exception as e:
        raise RuntimeError(f"[Embeddings] get_embeddings error: {e}")
