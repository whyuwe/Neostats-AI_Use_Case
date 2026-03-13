"""
app.py
───────
NeoStats AI Chatbot — Main Streamlit Application

Features:
- Multi-LLM support: Groq, OpenAI, Gemini (selectable in sidebar)
- RAG (Retrieval-Augmented Generation) with FAISS vector store
              
"""

import streamlit as st
import os, sys, tempfile

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config import (
    APP_TITLE, APP_ICON,
    DEFAULT_OPENAI_MODEL, DEFAULT_GROQ_MODEL, DEFAULT_GEMINI_MODEL,
    OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, TAVILY_API_KEY,
)
from models.llm import get_model
from utils.chat_utils import get_chat_response, get_rag_response, get_search_response
from utils.rag_utils import get_or_build_vectorstore, load_vectorstore


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Defaults ────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages":      [],
        "vectorstore":   None,
        "rag_enabled":   False,
        "search_enabled": False,
        "response_mode": "detailed",
        "provider":      "groq",
        "model_name":    DEFAULT_GROQ_MODEL,
        "system_prompt": "You are a helpful, knowledgeable AI assistant.",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/bot.png", width=64)
        st.title(APP_TITLE)
        st.divider()

        # ── Navigation ──
        page = st.radio("📌 Navigation", ["💬 Chat", "📄 RAG Setup", "ℹ️ Instructions"], index=0)
        st.divider()

        # ── Provider & Model ──
        st.subheader("🧠 Model Settings")
        provider = st.selectbox(
            "LLM Provider",
            ["groq", "openai", "gemini"],
            index=["groq", "openai", "gemini"].index(st.session_state.provider),
        )
        st.session_state.provider = provider

        model_options = {
            "groq":   ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
            "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        }
        model_name = st.selectbox("Model", model_options[provider])
        st.session_state.model_name = model_name
        st.divider()

        # ── Response Mode ──
        st.subheader("⚡ Response Mode")
        mode = st.radio(
            "Mode",
            ["detailed", "concise"],
            index=0 if st.session_state.response_mode == "detailed" else 1,
            format_func=lambda x: "📖 Detailed" if x == "detailed" else "⚡ Concise",
        )
        st.session_state.response_mode = mode
        st.divider()

        # ── Feature Toggles ──
        st.subheader("🔧 Features")
        st.session_state.rag_enabled    = st.toggle("📚 RAG (Document Q&A)", value=st.session_state.rag_enabled)
        st.session_state.search_enabled = st.toggle("🌐 Live Web Search",     value=st.session_state.search_enabled)

        if st.session_state.rag_enabled and st.session_state.search_enabled:
            st.warning("⚠️ RAG takes priority when both are enabled.")
        st.divider()

        # ── System Prompt ──
        st.subheader("🎭 System Prompt")
        st.session_state.system_prompt = st.text_area(
            "Customize AI behaviour",
            value=st.session_state.system_prompt,
            height=100,
        )
        st.divider()

        # ── API Key Status ──
        st.subheader("🔑 API Key Status")
        def key_status(label, key):
            icon = "✅" if key else "❌"
            st.markdown(f"{icon} **{label}**")
        key_status("Groq",    GROQ_API_KEY)
        key_status("OpenAI",  OPENAI_API_KEY)
        key_status("Gemini",  GEMINI_API_KEY)
        key_status("Tavily",  TAVILY_API_KEY)
        st.divider()

        # ── Clear Chat ──
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return page


# ── Chat Page ─────────────────────────────────────────────────────────────────

def chat_page():
    st.title(f"{APP_ICON} {APP_TITLE}")

    # Status bar
    col1, col2, col3 = st.columns(3)
    col1.metric("Provider", st.session_state.provider.upper())
    col2.metric("Mode",     st.session_state.response_mode.capitalize())
    col3.metric("RAG", "ON ✅" if st.session_state.rag_enabled and st.session_state.vectorstore
                               else ("ON ⚠️ No docs" if st.session_state.rag_enabled else "OFF"))

    st.divider()

    # Feature status notices
    if st.session_state.rag_enabled and not st.session_state.vectorstore:
        st.warning("📄 RAG is enabled but no documents are loaded. Go to **RAG Setup** in the sidebar.")

    if st.session_state.search_enabled:
        st.info("🌐 Live web search is active — queries will be augmented with real-time results.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask me anything..."):

        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Load LLM
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_model(st.session_state.provider, st.session_state.model_name)

                    # History excludes the just-added user message
                    history = st.session_state.messages[:-1]

                    # ── Route to correct chain ──
                    if st.session_state.rag_enabled and st.session_state.vectorstore:
                        response = get_rag_response(
                            llm=llm,
                            user_input=user_input,
                            history=history,
                            vectorstore=st.session_state.vectorstore,
                            response_mode=st.session_state.response_mode,
                        )
                        st.markdown(response)
                        st.caption("📚 *Answer generated using your uploaded documents (RAG)*")

                    elif st.session_state.search_enabled:
                        response = get_search_response(
                            llm=llm,
                            user_input=user_input,
                            history=history,
                            response_mode=st.session_state.response_mode,
                        )
                        st.markdown(response)
                        st.caption("🌐 *Answer augmented with live web search results*")

                    else:
                        response = get_chat_response(
                            llm=llm,
                            user_input=user_input,
                            history=history,
                            system_prompt=st.session_state.system_prompt,
                            response_mode=st.session_state.response_mode,
                        )
                        st.markdown(response)

                except Exception as e:
                    response = f"⚠️ Error: {str(e)}"
                    st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ── RAG Setup Page ────────────────────────────────────────────────────────────

def rag_setup_page():
    st.title("📄 RAG Document Setup")
    st.markdown(
        "Upload your documents below. The chatbot will use them to answer questions "
        "with relevant context retrieved via vector similarity search."
    )

    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )

    with col2:
        st.markdown("### 📊 Vectorstore Status")
        if st.session_state.vectorstore:
            st.success("✅ Vectorstore is ready")
        else:
            st.warning("⚠️ No vectorstore loaded")

        if st.button("🔄 Load Existing Vectorstore"):
            vs = load_vectorstore()
            if vs:
                st.session_state.vectorstore = vs
                st.session_state.rag_enabled = True
                st.success("✅ Existing vectorstore loaded!")
            else:
                st.error("No saved vectorstore found. Please upload documents first.")

    if uploaded_files:
        st.divider()
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.markdown(f"- 📄 `{f.name}` ({round(f.size/1024, 1)} KB)")

        if st.button("⚙️ Build Vectorstore from Uploaded Files", type="primary", use_container_width=True):
            with st.spinner("Processing documents and building vectorstore..."):
                try:
                    # Save uploads to temp dir
                    tmp_dir   = tempfile.mkdtemp()
                    tmp_paths = []
                    for uploaded in uploaded_files:
                        tmp_path = os.path.join(tmp_dir, uploaded.name)
                        with open(tmp_path, "wb") as f:
                            f.write(uploaded.read())
                        tmp_paths.append(tmp_path)

                    # Build vectorstore
                    vectorstore = get_or_build_vectorstore(tmp_paths)
                    st.session_state.vectorstore   = vectorstore
                    st.session_state.rag_enabled   = True
                    st.success(f"✅ Vectorstore built from {len(tmp_paths)} file(s)! RAG is now active.")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Error building vectorstore: {e}")

    st.divider()
    st.markdown("### 💡 How RAG Works")
    st.markdown("""
    1. **Upload** your PDF / TXT / DOCX files above
    2. **Build** — documents are split into chunks and embedded into a FAISS vector store
    3. **Ask** — your query is matched against the stored chunks via cosine similarity
    4. **Answer** — the top-k most relevant chunks are passed as context to the LLM
    """)


# ── Instructions Page ─────────────────────────────────────────────────────────

def instructions_page():
    st.title("ℹ️ Instructions & Setup")
    st.markdown("""
    ## 🚀 Quick Start

    ### 1. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

    ### 2. Set API keys in `config/config.py`
    ```python
    GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "your-key-here")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-key-here")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your-key-here")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "your-key-here")  # for web search
    ```
    Or set them as environment variables before running:
    ```bash
    export GROQ_API_KEY="your-key"
    streamlit run app.py
    ```

    ### 3. Run the app
    ```bash
    streamlit run app.py
    ```

    ---

    ## 🔑 Get API Keys

    | Provider | Link |
    |----------|------|
    | Groq (free) | https://console.groq.com/keys |
    | OpenAI | https://platform.openai.com/api-keys |
    | Google Gemini | https://aistudio.google.com/app/apikey |
    | Tavily (web search) | https://app.tavily.com |

    ---

    ## 🎛️ Features

    | Feature | How to use |
    |---------|------------|
    | **Multi-LLM** | Select provider + model in sidebar |
    | **Concise / Detailed** | Toggle response mode in sidebar |
    | **RAG** | Go to RAG Setup → upload docs → enable toggle |
    | **Web Search** | Enable 🌐 toggle in sidebar |
    | **Custom Persona** | Edit system prompt in sidebar |

    ---

    ## 🏗️ Architecture

    ```
    User Input
        │
        ├─ RAG enabled?  ──► retrieve_context() ──► get_rag_response()
        │                         (FAISS)
        ├─ Search enabled? ──► search_web() ──► get_search_response()
        │                      (Tavily / DDG)
        └─ Plain chat ──► get_chat_response()
                               │
                    LangChain LCEL Chain
                    ┌──────────────────────────────┐
                    │ ChatPromptTemplate            │
                    │  (system + history + input)   │
                    │        ↓                      │
                    │     LLM Model                 │
                    │        ↓                      │
                    │   StrOutputParser             │
                    └──────────────────────────────┘
    ```
    """)


# ── Main Router ───────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "💬 Chat":
        chat_page()
    elif page == "📄 RAG Setup":
        rag_setup_page()
    elif page == "ℹ️ Instructions":
        instructions_page()


if __name__ == "__main__":
    main()
