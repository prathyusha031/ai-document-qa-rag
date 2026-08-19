"""
config.py — Central configuration for the RAG application.

All settings are read from environment variables (loaded from a .env file
via python-dotenv) with sensible defaults, so the app works out of the box
and secrets are never hardcoded.
"""

import os

from dotenv import load_dotenv

# Load variables from the .env file in the project root (if present).
# If the file does not exist, the app simply uses the defaults below.
load_dotenv()


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to a default."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --- Gemini (LLM) settings -------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.7-flash").strip()

# --- Embedding model settings ----------------------------------------------
# Lightweight local model — good semantic similarity, no API cost.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

# --- Text chunking settings ------------------------------------------------
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 500)          # max characters per chunk
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 50)     # overlap between chunks

# --- Retrieval settings -----------------------------------------------------
TOP_K_DEFAULT: int = _get_int("TOP_K_DEFAULT", 5)      # chunks retrieved per question

# --- LLM generation settings ------------------------------------------------
# Low temperature = factual, less "creative" output (reduces hallucinations).
TEMPERATURE: float = _get_float("TEMPERATURE", 0.2)
MAX_OUTPUT_TOKENS: int = _get_int("MAX_OUTPUT_TOKENS", 1024)

# --- Vector database settings -----------------------------------------------
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db").strip()
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "document_chunks").strip()
