"""
utils/chat_utils.py
────────────────────
Core LangChain conversation chain utilities.

Builds a stateful chat chain using:
  - ChatPromptTemplate (system + history + human message)
  - LangChain Expression Language (LCEL) pipe syntax
  - ConversationBufferWindowMemory for trimmed history
  - StrOutputParser for clean string output

Response modes (concise / detailed) inject extra instruction
into the system prompt at chain build time.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import CONCISE_INSTRUCTION, DETAILED_INSTRUCTION, MAX_HISTORY
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_system_prompt(base_prompt: str, response_mode: str = "detailed") -> str:
    """
    Attach response-mode instruction to the base system prompt.
    response_mode: 'concise' | 'detailed'
    """
    mode_instruction = (
        CONCISE_INSTRUCTION if response_mode == "concise" else DETAILED_INSTRUCTION
    )
    return f"{base_prompt}\n\n{mode_instruction}".strip()


# ── LangChain Conversation Chain ───────────────────────────────────────────────

def build_chat_chain(llm, system_prompt: str):
    """
    Build a LangChain LCEL chat chain.

    Chain structure:
      messages (history + new human msg)
        → ChatPromptTemplate
        → LLM
        → StrOutputParser
        → str response

    Returns the compiled chain (Runnable).
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | llm | StrOutputParser()
        return chain
    except Exception as e:
        raise RuntimeError(f"[Chat] build_chat_chain error: {e}")


# ── History Formatter ──────────────────────────────────────────────────────────

def format_history(messages: list[dict]) -> list:
    """
    Convert Streamlit session message dicts to LangChain message objects.
    Trims to MAX_HISTORY turns to stay within context window.

    Input format:  [{"role": "user"|"assistant", "content": "..."}]
    Output format: [HumanMessage(...), AIMessage(...), ...]
    """
    try:
        # Keep only the last MAX_HISTORY messages (excluding the current user input)
        trimmed = messages[-(MAX_HISTORY * 2):]
        lc_messages = []
        for msg in trimmed:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        return lc_messages
    except Exception as e:
        raise RuntimeError(f"[Chat] format_history error: {e}")


# ── Main Invoke Function ───────────────────────────────────────────────────────

def get_chat_response(
    llm,
    user_input: str,
    history: list[dict],
    system_prompt: str,
    response_mode: str = "detailed",
) -> str:
    """
    End-to-end function: build chain → invoke → return string response.

    Parameters
    ----------
    llm           : LangChain chat model instance
    user_input    : The current user message string
    history       : Full Streamlit session message list (role/content dicts)
                    Should NOT include the current user_input yet.
    system_prompt : Base system prompt string
    response_mode : 'concise' | 'detailed'

    Returns
    -------
    str : The assistant's response text
    """
    try:
        full_system = build_system_prompt(system_prompt, response_mode)
        chain       = build_chat_chain(llm, full_system)
        lc_history  = format_history(history)

        response = chain.invoke({
            "history": lc_history,
            "input":   user_input,
        })
        return response

    except Exception as e:
        return f"⚠️ Error getting response: {str(e)}"


# ── RAG-Augmented Response ─────────────────────────────────────────────────────

def get_rag_response(
    llm,
    user_input: str,
    history: list[dict],
    vectorstore,
    response_mode: str = "detailed",
) -> str:
    """
    RAG-augmented chat: retrieves relevant context from vectorstore,
    then passes it along with the user query to the LLM via a chain.
    """
    try:
        from utils.rag_utils import build_rag_chain, retrieve_context

        context = retrieve_context(user_input, vectorstore)

        rag_system = (
            "You are a knowledgeable assistant. Use the retrieved document context "
            "below to answer the user's question. If the context does not contain "
            "enough information, say so and answer from your general knowledge.\n\n"
            f"Retrieved Context:\n{context}"
        )

        full_system = build_system_prompt(rag_system, response_mode)
        chain       = build_chat_chain(llm, full_system)
        lc_history  = format_history(history)

        response = chain.invoke({
            "history": lc_history,
            "input":   user_input,
        })
        return response

    except Exception as e:
        return f"⚠️ RAG error: {str(e)}"


# ── Search-Augmented Response ──────────────────────────────────────────────────

def get_search_response(
    llm,
    user_input: str,
    history: list[dict],
    response_mode: str = "detailed",
) -> str:
    """
    Web-search-augmented chat: fetches live results, injects into prompt,
    then runs through the LangChain chain.
    """
    try:
        from utils.search_utils import search_web

        search_results = search_web(user_input)

        search_system = (
            "You are a helpful assistant with access to live web search results. "
            "Use the search results below to answer the user's question accurately "
            "and cite the sources where relevant.\n\n"
            f"Live Web Search Results:\n{search_results}"
        )

        full_system = build_system_prompt(search_system, response_mode)
        chain       = build_chat_chain(llm, full_system)
        lc_history  = format_history(history)

        response = chain.invoke({
            "history": lc_history,
            "input":   user_input,
        })
        return response

    except Exception as e:
        return f"⚠️ Search error: {str(e)}"
