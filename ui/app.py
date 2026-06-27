"""Streamlit chat UI for the RAG Knowledge Assistant."""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from loguru import logger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.source-card {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
}
.metric-pill {
    display: inline-block;
    background: #e8f4f8;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 12px;
    color: #0066cc;
    margin-right: 6px;
}
.stChatMessage { border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_state():
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": str, "content": str, "sources": list}
    if "ingested" not in st.session_state:
        st.session_state.ingested = False
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "total_latency" not in st.session_state:
        st.session_state.total_latency = 0.0


def _init_rag(source=None, use_sample=False):
    """Initialise or re-initialise the RAG chain (cached in session state)."""
    from ingestion import run_ingestion_pipeline
    from chain.rag_chain import RAGChain

    with st.spinner("Building knowledge base…"):
        vector_store = run_ingestion_pipeline(source=source, use_sample=use_sample)
        st.session_state.rag_chain = RAGChain(vector_store)
        st.session_state.ingested = True


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("🔍 RAG Assistant")
        st.caption("Enterprise knowledge Q&A powered by RAG")

        st.divider()
        st.subheader("Knowledge base")

        # Sample docs button
        if st.button("Load sample documents", use_container_width=True):
            _init_rag(use_sample=True)
            st.success("Sample documents loaded!")

        # File uploader
        uploaded = st.file_uploader(
            "Upload your documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=False,
            help="PDF, DOCX, TXT, or Markdown files",
        )
        if uploaded and st.button("Ingest document", use_container_width=True):
            import tempfile
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            _init_rag(source=tmp_path)
            os.unlink(tmp_path)
            st.success(f"Ingested: {uploaded.name}")

        # URL ingestion
        url = st.text_input("Or enter a URL to scrape", placeholder="https://...")
        if url and st.button("Ingest URL", use_container_width=True):
            _init_rag(source=url)
            st.success(f"Ingested: {url}")

        st.divider()

        # Stats
        if st.session_state.query_count > 0:
            avg_latency = st.session_state.total_latency / st.session_state.query_count
            col1, col2 = st.columns(2)
            col1.metric("Queries", st.session_state.query_count)
            col2.metric("Avg latency", f"{avg_latency:.0f}ms")

        # Reset
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag_chain:
                st.session_state.rag_chain.reset_memory()
            from ingestion.embedder import clear_vector_store
            clear_vector_store()
            st.session_state.ingested = False
            st.rerun()

        st.divider()
        st.caption("Built with LangChain · ChromaDB · Streamlit")

        # Settings expander
        with st.expander("Settings"):
            st.caption(f"LLM: {os.getenv('LLM_PROVIDER', 'anthropic')}")
            st.caption(f"Embeddings: {os.getenv('EMBEDDING_PROVIDER', 'huggingface')}")
            st.caption(f"Vector DB: {os.getenv('VECTOR_DB', 'chroma')}")
            st.caption(f"Chunk size: {os.getenv('CHUNK_SIZE', '512')}")
            st.caption(f"Top-K: {os.getenv('TOP_K_RETRIEVAL', '5')}")


# ── Main chat ─────────────────────────────────────────────────────────────────

def render_chat():
    st.title("Enterprise Knowledge Assistant")
    st.caption("Ask questions about your documents. Answers are grounded in retrieved context.")

    if not st.session_state.ingested:
        st.info(
            "👈 Load sample documents or upload your own in the sidebar to get started.",
            icon="ℹ️",
        )
        return

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"], msg.get("latency_ms"))

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents…"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Stream tokens
            with st.spinner("Retrieving and generating…"):
                result = st.session_state.rag_chain.query(prompt)

            full_response = result["answer"]
            response_placeholder.markdown(full_response)

            sources = result.get("sources", [])
            latency = result.get("latency_ms", 0)

            _render_sources(sources, latency)

        # Save to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "latency_ms": latency,
            }
        )
        st.session_state.query_count += 1
        st.session_state.total_latency += latency


def _render_sources(sources: list, latency_ms: float | None = None):
    """Render source citations and performance pill below an answer."""
    if not sources and not latency_ms:
        return

    with st.expander(f"📄 Sources ({len(sources)})", expanded=False):
        for src in sources:
            topic = src.get("topic", "")
            source = src.get("source", "Unknown")
            page = src.get("page", "")
            preview = src.get("preview", "")

            label = topic or source
            meta = f"Source: {source}"
            if page:
                meta += f" | Page: {page}"

            st.markdown(
                f"""<div class="source-card">
                <strong>{label}</strong><br>
                <span style="color:#666;font-size:12px">{meta}</span><br>
                <em style="color:#444">{preview[:180]}…</em>
                </div>""",
                unsafe_allow_html=True,
            )

        if latency_ms:
            st.markdown(
                f'<span class="metric-pill">⚡ {latency_ms:.0f}ms</span>',
                unsafe_allow_html=True,
            )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _init_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
