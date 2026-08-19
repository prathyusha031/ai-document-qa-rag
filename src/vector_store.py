"""
vector_store.py — ChromaDB operations.

Responsibility:
    * Create a persistent local vector database (./chroma_db).
    * Store document chunks with their embeddings + metadata
      (source filename, page number, chunk id).
    * Search for the most similar chunks given a question embedding.

A new collection is created (or recreated) for each newly processed
document, so old documents never mix with new ones.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import chromadb

from . import config

logger = logging.getLogger(__name__)


class ChromaError(Exception):
    """Raised when ChromaDB operations fail (shown as a friendly message)."""


class VectorStore:
    """Persistent ChromaDB wrapper."""

    def __init__(self, persist_dir: str | None = None) -> None:
        self.persist_dir = persist_dir or config.CHROMA_DIR
        self._collection = None  # set by create_collection() / loaded on demand
        try:
            # PersistentClient keeps data on disk across app restarts.
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        except Exception as exc:
            logger.exception("Could not start ChromaDB client")
            raise ChromaError(
                "The vector database could not be started. Check that ./chroma_db "
                "is writable and try again."
            ) from exc

    def create_collection(self, name: str | None = None) -> None:
        """
        Create a fresh collection for a new document.

        If a collection with this name already exists (e.g. from a previous
        document), it is deleted first so old vectors never mix with new ones.
        """
        name = name or config.COLLECTION_NAME
        try:
            # Delete any previous collection with the same name.
            try:
                self._client.delete_collection(name)
            except Exception:
                pass  # collection did not exist yet — that is fine
            self._collection = self._client.create_collection(name=name)
            logger.info("Created collection '%s'", name)
        except Exception as exc:
            logger.exception("Failed to create collection '%s'", name)
            raise ChromaError(
                "The vector database collection could not be created. Please try again."
            ) from exc

    def add_chunks(
        self,
        chunks: List[dict],
        embeddings: List[list],
        source_name: str,
    ) -> None:
        """
        Store chunks with their embeddings and metadata.

        chunks:      [{"text": str, "page_number": int, "chunk_id": int}, ...]
        embeddings:  list of vectors, same order as chunks
        source_name: the uploaded file's name (stored as metadata)
        """
        if not chunks or embeddings is None or len(embeddings) == 0:
            raise ChromaError("No chunks to store in the vector database.")

        ids = [f"chunk_{c['chunk_id']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas: List[Dict] = [
            {
                "source": source_name,
                "page": c["page_number"],
                "chunk_id": c["chunk_id"],
            }
            for c in chunks
        ]

        try:
            # Convert numpy float32 values to plain Python floats — chromadb
            # rejects numpy scalars in embedding lists.
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=[[float(v) for v in row] for row in embeddings],
            )
            logger.info("Stored %s chunks in the vector database", len(chunks))
        except Exception as exc:
            logger.exception("Failed to store chunks in ChromaDB")
            raise ChromaError("Chunks could not be stored in the vector database.") from exc

    def _get_collection(self):
        """
        Return the active collection, loading it from disk if needed.

        The app creates a fresh RagPipeline for every question, so a new
        VectorStore instance may be asked to search a collection that was
        created by an earlier instance. This loads it from ChromaDB's
        persistent storage in that case.
        """
        if self._collection is None:
            self._collection = self._client.get_collection(config.COLLECTION_NAME)
        return self._collection

    def search(self, query_embedding: list, top_k: int = 5) -> List[dict]:
        """
        Return the top_k most similar chunks to the query embedding.

        Each result is a dict: {"text", "page", "source", "similarity"}
        (higher similarity = more relevant).
        """
        try:
            collection = self._get_collection()
            results = collection.query(
                query_embeddings=[[float(v) for v in query_embedding]],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.exception("Vector search failed")
            raise ChromaError("Searching the vector database failed. Please try again.") from exc

        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        hits: List[dict] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Chroma returns a distance; convert to a similarity score in [0, 1].
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            hits.append(
                {
                    "text": doc,
                    "page": int(meta.get("page", 0)),
                    "source": meta.get("source", ""),
                    "similarity": round(similarity, 4),
                }
            )
        return hits
