"""
rag_pipeline.py — ties retrieval + prompt construction + answer generation together.

The RAG flow implemented here:

    PDF bytes
      -> extract & chunk (pdf_processor)
      -> embed chunks (embeddings)
      -> store in ChromaDB (vector_store)

    User question
      -> embed question
      -> similarity search -> top K chunks
      -> build prompt (system instruction + retrieved context + question)
      -> Gemini LLM -> answer
      -> return answer + sources
"""

from __future__ import annotations

import logging
from typing import List

from . import config
from .embeddings import EmbeddingModel
from .llm import GeminiClient
from .pdf_processor import PdfProcessingError, process_pdf
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# --- Prompt engineering -----------------------------------------------------
# The system prompt is the anti-hallucination core of the app: the model is
# told to answer ONLY from the retrieved context and to admit when it cannot
# find the information.

SYSTEM_PROMPT = """\
You are an intelligent document assistant. You answer questions using ONLY the
retrieved context from the uploaded document.

Strict rules:
1. Base your answer exclusively on the retrieved document context.
2. Do NOT invent facts, figures, or details that are not in the context.
3. If the context does not contain the answer, say exactly:
   "I couldn't find this information in the uploaded document."
4. Keep answers concise but useful.
5. When possible, mention the page number(s) where the information appears.
6. Never mention that you are an AI model or that you used retrieval."""


def build_user_prompt(question: str, chunks: List[dict]) -> str:
    """
    Build the user prompt: retrieved context + the question.

    chunks: list of dicts with keys text, page, source, similarity.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Chunk {i} — Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)
    return (
        f"RETRIEVED DOCUMENT CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"ANSWER (using only the context above):"
    )


class RagPipeline:
    """High-level orchestrator for the document Q&A flow."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        llm: GeminiClient | None = None,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()
        self.llm = llm or GeminiClient()

    # -- Document ingestion --------------------------------------------------
    def process_document(self, pdf_bytes: bytes, source_name: str, **chunk_kwargs) -> dict:
        """
        Process an uploaded PDF: extract, chunk, embed, and store in ChromaDB.

        Returns a summary dict with file_name, pages, char_count, chunk_count.
        """
        try:
            result = process_pdf(
                pdf_bytes,
                chunk_size=chunk_kwargs.get("chunk_size", config.CHUNK_SIZE),
                chunk_overlap=chunk_kwargs.get("chunk_overlap", config.CHUNK_OVERLAP),
            )
        except PdfProcessingError:
            raise

        chunks = result["chunks"]
        if not chunks:
            raise PdfProcessingError(
                "No text chunks could be created from this PDF. It may be a scanned document."
            )

        # Embed every chunk and store them in a fresh collection.
        embeddings = self.embedding_model.embed_documents([c.text for c in chunks])
        self.vector_store.create_collection()
        self.vector_store.add_chunks(
            chunks=[{"text": c.text, "page_number": c.page_number, "chunk_id": c.chunk_id} for c in chunks],
            embeddings=embeddings,
            source_name=source_name,
        )

        return {
            "file_name": source_name,
            "pages": len(result["pages"]),
            "char_count": len(result["text"]),
            "chunk_count": len(chunks),
        }

    # -- Question answering --------------------------------------------------
    def ask(self, question: str, top_k: int = 5) -> dict:
        """
        Answer a question about the processed document.

        Returns a dict with: answer, sources (list of page numbers), chunks
        (the retrieved context with similarity scores).
        """
        if not question or not question.strip():
            raise ValueError("Question is empty.")

        # 1. Embed the question and find the most similar chunks.
        query_embedding = self.embedding_model.embed_query(question.strip())
        chunks = self.vector_store.search(query_embedding, top_k=top_k)
        if not chunks:
            raise ValueError("No relevant content found in the vector database.")

        # 2. Build the prompt and ask the LLM.
        user_prompt = build_user_prompt(question.strip(), chunks)
        answer = self.llm.generate_answer(SYSTEM_PROMPT, user_prompt)

        # 3. Collect the sources (unique page numbers, in order).
        sources: List[int] = []
        for chunk in chunks:
            if chunk["page"] not in sources:
                sources.append(chunk["page"])

        return {"answer": answer, "sources": sources, "chunks": chunks}
