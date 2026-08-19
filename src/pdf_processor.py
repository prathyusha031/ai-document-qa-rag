"""
pdf_processor.py — PDF text extraction, cleaning, and chunking.

Responsibility:
    1. Read the uploaded PDF page by page (using pypdf).
    2. Clean the extracted text (normalize whitespace).
    3. Split the text into small, meaningful chunks for embedding.
    4. Keep track of which page each chunk came from (for sources).

The module never executes content from the PDF — it only reads text.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PdfProcessingError(Exception):
    """Raised when a PDF cannot be processed (shown to the user as a friendly message)."""


@dataclass
class PageText:
    """Text extracted from a single page of the PDF."""
    page_number: int   # 1-based page number (as humans count pages)
    text: str


@dataclass
class Chunk:
    """A single chunk of text, ready to be embedded and stored."""
    text: str
    page_number: int
    chunk_id: int


def extract_text_from_pdf(pdf_bytes: bytes) -> List[PageText]:
    """
    Extract text page-by-page from raw PDF bytes.

    Raises PdfProcessingError with a friendly message when the PDF
    is invalid, empty, or contains no extractable text (e.g. a scan).
    """
    if not pdf_bytes:
        raise PdfProcessingError("The uploaded file is empty. Please upload a valid PDF.")

    try:
        # pypdf 6 needs a seekable stream, so wrap the bytes in BytesIO.
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:  # invalid / corrupted PDF
        logger.warning("Could not parse PDF: %s", exc)
        raise PdfProcessingError(
            "The file does not appear to be a valid PDF. Please upload a PDF document."
        ) from exc

    if len(reader.pages) == 0:
        raise PdfProcessingError("The PDF has no pages. Please upload a valid document.")

    pages: List[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed to extract text from page %s: %s", index, exc)
            text = ""
        pages.append(PageText(page_number=index, text=clean_text(text)))

    # A scanned / image-only PDF has no text layer, so nothing was extracted.
    if all(not page.text for page in pages):
        raise PdfProcessingError(
            "No text could be extracted from this PDF. It may be a scanned or "
            "image-only document — OCR is not supported in this version."
        )

    return pages


def clean_text(text: str) -> str:
    """Normalize whitespace: collapse repeated spaces/newlines into single spaces."""
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split cleaned text into overlapping chunks of ~chunk_size characters.

    This simple recursive splitter keeps words intact: it finds the last
    sentence/space boundary before the limit and breaks there, sliding
    forward by chunk_size - chunk_overlap characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Try to break at a sentence end or word boundary near the limit.
            window = text[start:end]
            last_sentence = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if last_sentence > chunk_size * 0.5:
                end = start + last_sentence + 2
            else:
                last_space = window.rfind(" ")
                if last_space > chunk_size * 0.5:
                    end = start + last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break
        start = end - chunk_overlap

    return chunks


def process_pdf(
    pdf_bytes: bytes,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    """
    Full PDF processing pipeline: extract -> clean -> chunk.

    Returns a dict with:
        file_name   : str        (set later by the caller)
        pages       : List[PageText]
        text        : str        (full cleaned text, for the character count)
        chunks      : List[Chunk]
    """
    pages = extract_text_from_pdf(pdf_bytes)
    full_text = " ".join(page.text for page in pages)

    # Build the chunks with page numbers attached to each one.
    chunks: List[Chunk] = []
    chunk_id = 0
    for page in pages:
        for piece in chunk_text(page.text, chunk_size, chunk_overlap):
            chunks.append(Chunk(text=piece, page_number=page.page_number, chunk_id=chunk_id))
            chunk_id += 1

    return {
        "pages": pages,
        "text": full_text,
        "chunks": chunks,
    }
