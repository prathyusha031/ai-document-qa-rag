"""
app.py — Streamlit user interface for the AI Document Q&A (RAG) application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src import config
from src.embeddings import EmbeddingError
from src.llm import GeminiClient, LLMError
from src.pdf_processor import PdfProcessingError
from src.rag_pipeline import RagPipeline
from src.vector_store import ChromaError

# Logging for developers (errors still show friendly messages to users).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

st.set_page_config(page_title="Intelligent Document Q&A Assistant", page_icon="📄", layout="centered")

# ---------------------------------------------------------------------------
# Custom CSS — modern, polished UI (purely visual; no logic changes)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global — Purple theme ─────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #734F96;
        color: #f0eaf8;
    }
    html, body, [data-testid="stAppViewBlockContainer"] {
        background-color: #734F96;
    }

    /* ── Header ────────────────────────────────────────────────────── */
    h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
    .app-header { text-align: center; padding: 1.2rem 0 0.2rem 0; }
    .app-header h1 {
        font-size: 2.2rem !important;
        margin-bottom: 0.15rem;
        background: linear-gradient(135deg, #b49bff 0%, #c9aaff 50%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .app-header p { font-size: 0.95rem; color: #c9b4f0; margin-top: 0; }

    /* ── Sidebar ───────────────────────────────────────────────────── */
    [data-testid="stSidebar"] { background-color: #5d3f7e; }
    [data-testid="stSidebar"] h3 { font-weight: 600; color: #f0eaf8; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #ddd0f0 !important;
    }

    /* ── Buttons ───────────────────────────────────────────────────── */
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #667eea 0%, #9b6dd7 100%);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.01em;
        color: #fff !important;
        box-shadow: 0 4px 14px rgba(102,126,234,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(155,109,215,0.45);
    }
    button[data-testid="stBaseButton-primary"]:active { transform: translateY(0); }
    button[data-testid="stBaseButton"] {
        border: 1px solid rgba(200,180,255,0.2) !important;
        color: #c4badb !important;
        background: rgba(200,180,255,0.06);
        border-radius: 10px;
    }
    button[data-testid="stBaseButton"]:hover {
        background: rgba(200,180,255,0.12);
    }

    /* ── Metric cards ──────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.20);
        border-radius: 14px;
        padding: 18px 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.20);
        transition: box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.30); }
    [data-testid="stMetric"] label { font-weight: 500 !important; font-size: 0.82rem !important; color: #d0c4e8 !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-weight: 700 !important; color: #ffffff !important; }

    /* ── Chat messages ─────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.08);
    }
    /* user message accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-left: 3px solid #ffffff;
    }
    /* assistant message accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid #c9b4f0;
    }

    /* ── Expander ──────────────────────────────────────────────────── */
    .stExpander {
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        background: rgba(255,255,255,0.06);
    }
    .stExpander summary { color: #e0d6f0 !important; }

    /* ── Alerts (success / error / info / warning) ─────────────────── */
    [data-testid="stAlert"] { border-radius: 10px; }
    [data-testid="stAlert"] p, [data-testid="stAlert"] span { color: #f0eaf8 !important; }

    /* ── Section divider ───────────────────────────────────────────── */
    .section-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #b49bff, transparent);
        margin: 1.2rem 0;
    }

    /* ── Footer ────────────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: #b8a8d4;
        font-size: 0.8rem;
    }
    .app-footer span { font-weight: 600; color: #e0d6f0; }

    /* ── Chat input polish ─────────────────────────────────────────── */
    [data-testid="stChatInput"] { border-radius: 14px; }

    /* ── Text & markdown ───────────────────────────────────────────── */
    .stMarkdown p, .stMarkdown li { color: #f0eaf8; }
    h2, h3, .stSubheader { color: #f0eaf8 !important; }
    .stCaption, [data-testid="stCaptionContainer"] span { color: #b8a8d4 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header">'
    "<h1>📄 Intelligent Document Q&amp;A Assistant</h1>"
    "<p>Ask questions about your PDF using Retrieval-Augmented Generation (RAG)</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state (chat history lives here for the whole session)
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # list of {"role", "content", "sources", "chunks"}
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "document_info" not in st.session_state:
    st.session_state.document_info = None


def _render_sources(result: dict) -> None:
    """Show the sources (page numbers) and an expander with the retrieved chunks."""
    sources = result.get("sources", [])
    chunks = result.get("chunks", [])

    if sources:
        page_list = ", ".join(f"Page {p}" for p in sources)
        st.markdown(f"**Sources:** {page_list}")

    if chunks:
        with st.expander("View retrieved context"):
            for i, chunk in enumerate(chunks, start=1):
                st.markdown(
                    f"**Chunk {i}** — Page {chunk['page']} (similarity "
                    f"{chunk['similarity']:.2f})"
                )
                st.write(chunk["text"])
                st.divider()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Gemini API status
    llm = GeminiClient()
    if llm.is_configured():
        st.success(f"✅ Gemini API connected\nModel: `{config.GEMINI_MODEL}`")
    else:
        st.warning("⚠️ Gemini API key missing\nAdd `GEMINI_API_KEY` to your `.env` file.")

    st.divider()

    # PDF upload
    uploaded_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])

    # Process button
    process_clicked = st.button("⚙️ Process Document", type="primary", use_container_width=True)

    # Retrieval settings
    top_k = st.slider("🔍 Retrieved chunks (K)", min_value=3, max_value=8, value=config.TOP_K_DEFAULT)

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    with st.expander("ℹ️ About this app"):
        st.markdown(
            """
            **AI Document Q&A (RAG)**

            Upload a PDF, then ask questions in plain English.

            Pipeline: PDF → text → chunks → embeddings → ChromaDB
            → retrieval → Gemini → answer + sources.

            Embeddings run locally (`all-MiniLM-L6-v2`); answers come from
            Google's Gemini API (requires `GEMINI_API_KEY`).
            """
        )

# ---------------------------------------------------------------------------
# Process the document when the button is clicked
# ---------------------------------------------------------------------------
if uploaded_file is not None and process_clicked:
    with st.spinner("Processing document — extracting text, chunking, embedding, storing in vector DB..."):
        try:
            pdf_bytes = uploaded_file.getvalue()
            pipeline = RagPipeline()
            info = pipeline.process_document(pdf_bytes, source_name=uploaded_file.name)
            st.session_state.document_processed = True
            st.session_state.document_info = info
            st.session_state.chat_history = []  # fresh doc -> fresh chat
            st.success("✅ Document processed successfully!")
        except PdfProcessingError as exc:
            st.error(f"❌ {exc}")
            logger.warning("PDF processing error: %s", exc)
        except (EmbeddingError, ChromaError) as exc:
            st.error(f"❌ {exc}")
            logger.error("Pipeline infrastructure error: %s", exc)
        except Exception as exc:  # unexpected error: log detail, hide traceback
            st.error("❌ Something went wrong while processing the document. Please try again.")
            logger.exception("Unexpected error during document processing")

# ---------------------------------------------------------------------------
# Document summary (after processing)
# ---------------------------------------------------------------------------
if st.session_state.document_processed and st.session_state.document_info:
    info = st.session_state.document_info
    st.subheader("📊 Document summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("File", info["file_name"], help="Uploaded file name")
    col2.metric("Pages", info["pages"])
    col3.metric("Characters", f"{info['char_count']:,}")
    col4.metric("Chunks", info["chunk_count"])
    st.info("You can now ask questions about the document below. 👇")
elif uploaded_file is not None and not st.session_state.document_processed:
    st.info("Click **⚙️ Process Document** in the sidebar to index this PDF.")

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.subheader("💬 Ask a question")

# Show the conversation so far
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            _render_sources(message)

# Question input (disabled until a document has been processed)
chat_disabled = not st.session_state.document_processed
question = st.chat_input("Ask a question about your document...", disabled=chat_disabled)

if question:
    question = question.strip()
    if not question:
        st.warning("Please type a question first.")
    elif not st.session_state.document_processed:
        st.warning("Please upload and process a PDF document first.")
    else:
        # Show the user's question immediately
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Generate the answer
        with st.chat_message("assistant"):
            with st.spinner("Searching the document and generating an answer..."):
                try:
                    pipeline = RagPipeline()
                    result = pipeline.ask(question, top_k=top_k)
                    st.markdown(result["answer"])
                    _render_sources(result)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "chunks": result["chunks"],
                        }
                    )
                except LLMError as exc:
                    st.error(f"❌ {exc}")
                    logger.warning("LLM error: %s", exc)
                except (EmbeddingError, ChromaError) as exc:
                    st.error(f"❌ {exc}")
                    logger.error("Pipeline infrastructure error: %s", exc)
                except Exception as exc:
                    st.error("❌ Something went wrong while answering. Please try again.")
                    logger.exception("Unexpected error while answering")

if chat_disabled:
    st.caption("👆 Upload and process a PDF first to enable the chat.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-footer">Built with <span>Streamlit</span>, PyPDF, '
    "sentence-transformers, ChromaDB &amp; Google Gemini</div>",
    unsafe_allow_html=True,
)
