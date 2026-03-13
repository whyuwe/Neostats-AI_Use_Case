"""
utils/rag_utils.py
───────────────────
RAG (Retrieval-Augmented Generation) pipeline utilities.

Responsibilities:
  - Load and parse uploaded documents (PDF, TXT, DOCX)
  - Split documents into overlapping chunks
  - Build / load a FAISS vector store
  - Retrieve top-k relevant chunks for a query
  - Build a LangChain RAG chain using retrieved context
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, VECTORSTORE_DIR
from models.embeddings import get_embeddings


# ── Document Loading ──────────────────────────────────────────────────────────

def load_documents(file_paths: list[str]) -> list:
    """
    Load documents from a list of file paths.
    Supports: .pdf, .txt, .docx
    Returns a list of LangChain Document objects.
    """
    try:
        from langchain_community.document_loaders import (
            PyPDFLoader, TextLoader, Docx2txtLoader
        )
        docs = []
        for path in file_paths:
            ext = os.path.splitext(path)[-1].lower()
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(path)
                elif ext == ".txt":
                    loader = TextLoader(path, encoding="utf-8")
                elif ext == ".docx":
                    loader = Docx2txtLoader(path)
                else:
                    print(f"[RAG] Unsupported file type: {ext} — skipping {path}")
                    continue
                docs.extend(loader.load())
            except Exception as e:
                print(f"[RAG] Error loading {path}: {e}")
        return docs
    except Exception as e:
        raise RuntimeError(f"[RAG] load_documents error: {e}")


# ── Text Splitting ─────────────────────────────────────────────────────────────

def split_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks for embedding.
    chunk_size and chunk_overlap come from config.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        print(f"[RAG] Split {len(documents)} doc(s) into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        raise RuntimeError(f"[RAG] split_documents error: {e}")


# ── Vector Store ──────────────────────────────────────────────────────────────

def build_vectorstore(chunks: list, persist: bool = True):
    """
    Build a FAISS vector store from document chunks.
    Optionally persists to disk at VECTORSTORE_DIR.
    Returns the FAISS vectorstore object.
    """
    try:
        from langchain_community.vectorstores import FAISS
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        if persist:
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            vectorstore.save_local(VECTORSTORE_DIR)
            print(f"[RAG] Vectorstore saved to '{VECTORSTORE_DIR}'.")
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"[RAG] build_vectorstore error: {e}")


def load_vectorstore():
    """
    Load a previously persisted FAISS vector store from disk.
    Returns None if no store exists yet.
    """
    try:
        from langchain_community.vectorstores import FAISS
        if not os.path.exists(VECTORSTORE_DIR):
            return None
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )
        print(f"[RAG] Vectorstore loaded from '{VECTORSTORE_DIR}'.")
        return vectorstore
    except Exception as e:
        print(f"[RAG] load_vectorstore error: {e}")
        return None


def get_or_build_vectorstore(file_paths: list[str]):
    """
    Full pipeline: load docs → split → build vectorstore.
    Always rebuilds when new files are provided.
    """
    try:
        docs   = load_documents(file_paths)
        chunks = split_documents(docs)
        return build_vectorstore(chunks, persist=True)
    except Exception as e:
        raise RuntimeError(f"[RAG] get_or_build_vectorstore error: {e}")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(query: str, vectorstore) -> str:
    """
    Retrieve top-k relevant chunks from the vectorstore for a given query.
    Returns a single concatenated context string.
    """
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
        results   = retriever.invoke(query)
        context   = "\n\n---\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        raise RuntimeError(f"[RAG] retrieve_context error: {e}")


# ── RAG Chain ─────────────────────────────────────────────────────────────────

def build_rag_chain(llm, vectorstore):
    """
    Build a LangChain RAG chain:
      query → retriever → prompt (context + question) → LLM → answer

    Uses LangChain Expression Language (LCEL) pipe syntax.
    """
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Use the following retrieved context to answer "
             "the user's question accurately. If the context does not contain the answer, "
             "say so honestly.\n\nContext:\n{context}"),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        raise RuntimeError(f"[RAG] build_rag_chain error: {e}")
