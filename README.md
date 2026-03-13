# NeoStats AI Chatbot 🤖

A production-ready, multi-provider AI chatbot built with **Streamlit** and **LangChain**,
featuring RAG (document Q&A), live web search, and configurable response modes.

---

## 📁 Project Structure

```
neostats_chatbot/
├── config/
│   ├── __init__.py
│   └── config.py           ← All API keys & app settings (env vars)
│
├── models/
│   ├── __init__.py
│   ├── llm.py              ← LLM factory (OpenAI / Groq / Gemini)
│   └── embeddings.py       ← Embedding models for RAG
│
├── utils/
│   ├── __init__.py
│   ├── chat_utils.py       ← LangChain LCEL conversation chains
│   ├── rag_utils.py        ← Document loading, chunking, FAISS vectorstore
│   └── search_utils.py     ← Tavily / DuckDuckGo web search
│
├── data/
│   └── vectorstore/        ← Persisted FAISS index (auto-created)
│
├── app.py                  ← Main Streamlit application
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API keys
Edit `config/config.py` or set environment variables:
```bash
export GROQ_API_KEY="your-groq-key"
export OPENAI_API_KEY="your-openai-key"      # optional
export GEMINI_API_KEY="your-gemini-key"      # optional
export TAVILY_API_KEY="your-tavily-key"      # optional, for web search
```

### 3. Run
```bash
streamlit run app.py
```

---

## ✨ Features

### 🧠 Multi-Provider LLM Support
| Provider | Models |
|----------|--------|
| **Groq** (recommended, free tier) | llama3-70b, llama3-8b, mixtral-8x7b, gemma2-9b |
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-3.5-turbo |
| **Google Gemini** | gemini-1.5-flash, gemini-1.5-pro |

### 📚 RAG — Document Q&A
- Upload PDF, TXT, DOCX files
- Documents are chunked and embedded using `all-MiniLM-L6-v2` (local, free)
- FAISS vector store with cosine similarity retrieval
- Top-k relevant chunks injected as context into the LLM prompt

### 🌐 Live Web Search
- Primary: **Tavily Search API** (clean, LLM-optimised results)
- Fallback: **DuckDuckGo** (no API key needed)
- Search results injected into LangChain prompt before LLM call

### ⚡ Response Modes
- **Detailed**: Thorough, structured responses with examples
- **Concise**: 2-3 sentence summaries

### 🔗 LangChain LCEL Chain
All conversations go through a proper LangChain chain:
```
ChatPromptTemplate (system + history + input)
    → LLM (OpenAI / Groq / Gemini)
    → StrOutputParser
    → str response
```

---

## 🏗️ Architecture

```
User Input
    │
    ├─ RAG enabled + docs loaded
    │       └─► retrieve_context (FAISS) ─► get_rag_response()
    │
    ├─ Web search enabled
    │       └─► search_web (Tavily/DDG) ─► get_search_response()
    │
    └─ Plain chat
            └─► get_chat_response()
                        │
              LangChain LCEL Chain
              ┌────────────────────────────┐
              │ ChatPromptTemplate         │
              │  system + history + input  │
              │          ↓                 │
              │       LLM Model            │
              │          ↓                 │
              │   StrOutputParser → str    │
              └────────────────────────────┘
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repo → select `app.py`
4. Add secrets in Streamlit Cloud dashboard:
   ```
   GROQ_API_KEY = "..."
   OPENAI_API_KEY = "..."
   TAVILY_API_KEY = "..."
   ```
5. Deploy 🚀
