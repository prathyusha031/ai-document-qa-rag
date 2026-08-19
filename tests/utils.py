"""
Test helpers.

    * make_sample_pdf()  — builds a real, minimal PDF with a text layer,
                           without needing reportlab/fpdf (raw PDF syntax).
    * FakeEmbeddingModel — deterministic, hash-based embeddings so tests are
                           fast and need no model download.
    * FakeLLM            — canned answer, records the prompt it received.
"""

from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Sample document (mirrors data/sample_document.pdf)
# ---------------------------------------------------------------------------
SAMPLE_PAGES: List[str] = [
    "Welcome to the Pet Care Guide. This guide explains how to keep your pets "
    "healthy and happy. It covers feeding, exercise, hygiene, and general wellbeing.",

    "Feeding: Cats need protein-rich food to stay healthy. Provide fresh water "
    "every day. Feed adult cats twice a day and follow the portions on the food package.",

    "Exercise: Dogs need daily walks to stay fit and mentally stimulated. "
    "A walk of thirty minutes each day is recommended for most breeds.",

    "Hygiene: Bathe your pets regularly and keep their living area clean. "
    "Visit a veterinarian at least once a year for a health check.",
]


def make_sample_pdf(pages_text: List[str] | None = None) -> bytes:
    """
    Build a minimal but valid PDF where every page has real extractable text.

    The PDF is assembled by hand (correct object table + xref) so the tests
    have no dependency on reportlab/fpdf.
    """
    pages_text = pages_text if pages_text is not None else SAMPLE_PAGES
    n = len(pages_text)

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict = {}

    def register(num: int, body: bytes) -> None:
        nonlocal out  # mutate the enclosing bytearray
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode()
        out += body
        out += b"\nendobj\n"

    catalog_num, pages_num = 1, 2
    page_nums = [3 + 2 * i for i in range(n)]
    content_nums = [4 + 2 * i for i in range(n)]
    font_num = 3 + 2 * n

    register(catalog_num, b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{num} 0 R" for num in page_nums)
    register(pages_num, f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())

    for i, text in enumerate(pages_text):
        register(
            page_nums[i],
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_nums[i]} 0 R "
                f"/Resources << /Font << /F1 {font_num} 0 R >> >> >>"
            ).encode(),
        )
        # Escape PDF string special characters and collapse newlines.
        escaped = (
            text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ).replace("\n", " ")
        stream = f"BT /F1 12 Tf 50 720 Td ({escaped}) Tj ET".encode()
        register(
            content_nums[i],
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        )

    register(font_num, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    xref_pos = len(out)
    total = font_num + 1
    out += b"xref\n"
    out += f"0 {total}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, total):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


# ---------------------------------------------------------------------------
# Fake embedding model (deterministic, no download needed)
# ---------------------------------------------------------------------------
class FakeEmbeddingModel:
    """Hash-based bag-of-words embeddings: similar texts get similar vectors."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            # Double hashing (like a Bloom filter): two independent dims per
            # token, so accidental collisions are far less likely.
            h1 = int(hashlib.md5(("a:" + token).encode("utf-8")).hexdigest(), 16)
            h2 = int(hashlib.md5(("b:" + token).encode("utf-8")).hexdigest(), 16)
            vec[h1 % self.dim] += 1.0
            vec[h2 % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.array([self._embed(t) for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


# ---------------------------------------------------------------------------
# Fake LLM (records the prompt, returns a canned answer)
# ---------------------------------------------------------------------------
class FakeLLM:
    def __init__(self, answer: str = "This is a canned test answer.") -> None:
        self.answer = answer
        self.last_system_prompt = None
        self.last_user_prompt = None

    def is_configured(self) -> bool:
        return True

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.answer
