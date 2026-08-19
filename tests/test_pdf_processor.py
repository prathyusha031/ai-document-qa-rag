"""Tests for src/pdf_processor.py — extraction, cleaning, and chunking."""

import pytest

from src.pdf_processor import (
    PdfProcessingError,
    chunk_text,
    clean_text,
    extract_text_from_pdf,
    process_pdf,
)
from tests.utils import SAMPLE_PAGES, make_sample_pdf


def _blank_pdf() -> bytes:
    """A valid PDF with NO text layer (simulates a scanned document)."""
    return make_sample_pdf([""])


def test_extract_text_from_valid_pdf():
    pages = extract_text_from_pdf(make_sample_pdf())
    assert len(pages) == len(SAMPLE_PAGES)
    assert pages[0].page_number == 1
    assert "Pet Care Guide" in pages[0].text
    # Page numbers are preserved and 1-based.
    assert [p.page_number for p in pages] == list(range(1, len(SAMPLE_PAGES) + 1))


def test_extract_raises_on_empty_bytes():
    with pytest.raises(PdfProcessingError):
        extract_text_from_pdf(b"")


def test_extract_raises_on_invalid_pdf():
    with pytest.raises(PdfProcessingError):
        extract_text_from_pdf(b"this is definitely not a pdf file")


def test_extract_raises_on_scanned_pdf():
    """A PDF without a text layer is reported as scanned/image-only."""
    with pytest.raises(PdfProcessingError, match="No text could be extracted"):
        extract_text_from_pdf(_blank_pdf())


def test_clean_text_normalizes_whitespace():
    assert clean_text("Hello    world.\n\n  Next line.") == "Hello world. Next line."
    assert clean_text("   ") == ""


def test_chunk_text_splits_long_text():
    long_text = " ".join(["word"] * 2000)  # ~10k chars
    chunks = chunk_text(long_text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 510 for c in chunks)  # chunk_size + a little slack


def test_chunk_text_short_text_is_single_chunk():
    assert chunk_text("A short sentence.", chunk_size=500) == ["A short sentence."]


def test_chunk_text_empty_text_returns_empty_list():
    assert chunk_text("", chunk_size=500) == []


def test_chunk_text_keeps_words_intact():
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text(text, chunk_size=12, chunk_overlap=0)
    for chunk in chunks:
        # No chunk should contain a half-word (starts/ends cleanly).
        assert chunk == chunk.strip()


def test_chunk_text_overlap_between_chunks():
    text = " ".join(["word"] * 300)  # ~1500 chars
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=60)
    if len(chunks) > 1:
        # Consecutive chunks share words (overlap slides the window forward).
        shared = set(chunks[0].split()) & set(chunks[1].split())
        assert shared


def test_process_pdf_returns_summary_with_chunks():
    result = process_pdf(make_sample_pdf(), chunk_size=300, chunk_overlap=30)
    assert result["text"]
    assert len(result["pages"]) == len(SAMPLE_PAGES)
    assert len(result["chunks"]) >= 1
    # Every chunk remembers which page it came from.
    for chunk in result["chunks"]:
        assert 1 <= chunk.page_number <= len(SAMPLE_PAGES)
