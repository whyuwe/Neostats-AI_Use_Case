"""
utils/search_utils.py
──────────────────────
Live web search integration using Tavily Search API.

Tavily is purpose-built for LLM agents — it returns clean, summarised
results rather than raw HTML, making it ideal for feeding into a prompt.

Fallback: DuckDuckGo (no API key required) when Tavily is unavailable.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import TAVILY_API_KEY


def search_web_tavily(query: str, max_results: int = 4) -> str:
    """
    Perform a live web search using Tavily Search API.
    Returns a formatted string of results to inject into the LLM prompt.
    Requires: TAVILY_API_KEY in config.
    """
    try:
        from tavily import TavilyClient
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is not set in config/config.py")

        client  = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query=query, max_results=max_results)

        formatted = []
        for i, r in enumerate(results.get("results", []), 1):
            title   = r.get("title", "No title")
            url     = r.get("url", "")
            content = r.get("content", "")
            formatted.append(f"**[{i}] {title}**\nURL: {url}\n{content}")

        return "\n\n".join(formatted) if formatted else "No results found."

    except Exception as e:
        return f"[Search] Tavily search error: {e}"


def search_web_duckduckgo(query: str, max_results: int = 4) -> str:
    """
    Fallback web search using DuckDuckGo (no API key needed).
    Uses duckduckgo-search library.
    """
    try:
        from duckduckgo_search import DDGS
        results   = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title   = r.get("title", "")
                href    = r.get("href", "")
                body    = r.get("body", "")
                results.append(f"**{title}**\nURL: {href}\n{body}")
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"[Search] DuckDuckGo search error: {e}"


def search_web(query: str, max_results: int = 4) -> str:
    """
    Unified search entry point.
    Uses Tavily if API key is available, otherwise falls back to DuckDuckGo.
    """
    try:
        if TAVILY_API_KEY:
            return search_web_tavily(query, max_results)
        else:
            print("[Search] No Tavily key found — using DuckDuckGo fallback.")
            return search_web_duckduckgo(query, max_results)
    except Exception as e:
        return f"[Search] search_web error: {e}"


def build_search_augmented_prompt(query: str, search_results: str) -> str:
    """
    Inject web search results into a prompt string for the LLM.
    Returns a formatted system + user prompt string.
    """
    return (
        f"Use the following real-time web search results to answer the question.\n"
        f"If the results don't contain the answer, say so honestly.\n\n"
        f"Search Results:\n{search_results}\n\n"
        f"Question: {query}"
    )
