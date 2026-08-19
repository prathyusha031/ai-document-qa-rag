"""Tests for the RAG components: embeddings, vector store, and pipeline."""

import pytest

from src.embeddings import EmbeddingModel
from src.llm import GeminiClient, LLMError, MISSING_KEY_MESSAGE
from src.rag_pipeline import RagPipeline, build_user_prompt
from src.vector_store import VectorStore
from tests.utils import FakeEmbeddingModel, FakeLLM, make_sample_pdf


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def test_fake_embedding_model_produces_normalised_vectors():
    model = FakeEmbeddingModel(dim=384)
    vectors = model.embed_documents(["cats eat fish", "dogs run fast"])
    assert vectors.shape == (2, 384)
    # Vectors are unit-length (cosine-ready).
    norms = (vectors ** 2).sum(axis=1) ** 0.5
    assert norms[0] == pytest.approx(1.0, abs=1e-5)


def test_real_embedding_model_loads_lazily_and_embeds():
    """Uses the real sentence-transformers model (downloads on first run)."""
    model = EmbeddingModel()
    vec = model.embed_query("What should cats eat?")
    assert vec.shape == (384,)
    assert float((vec ** 2).sum() ** 0.5) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def test_vector_store_create_and_add(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    store.create_collection("test_coll")
    store.add_chunks(
        chunks=[
            {"text": "cats eat fish", "page_number": 2, "chunk_id": 0},
            {"text": "dogs run fast", "page_number": 3, "chunk_id": 1},
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        source_name="guide.pdf",
    )
    results = store.search([1.0, 0.0], top_k=2)
    assert results[0]["text"] == "cats eat fish"
    assert results[0]["page"] == 2
    assert results[0]["source"] == "guide.pdf"


def test_vector_store_search_ranks_by_similarity(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    store.create_collection("test_coll")
    store.add_chunks(
        chunks=[
            {"text": "feeding cats", "page_number": 2, "chunk_id": 0},
            {"text": "walking dogs", "page_number": 3, "chunk_id": 1},
            {"text": "cleaning cages", "page_number": 4, "chunk_id": 2},
        ],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        source_name="guide.pdf",
    )
    results = store.search([0.9, 0.1, 0.0], top_k=3)
    assert results[0]["page"] == 2            # most similar first
    assert results[0]["similarity"] > results[1]["similarity"]


def test_vector_store_recreate_collection_removes_old_data(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    store.create_collection("test_coll")
    store.add_chunks(
        chunks=[{"text": "old document content", "page_number": 1, "chunk_id": 0}],
        embeddings=[[1.0, 0.0]],
        source_name="old.pdf",
    )
    # A new document re-creates the collection -> old vectors must be gone.
    store.create_collection("test_coll")
    store.add_chunks(
        chunks=[{"text": "new document content", "page_number": 1, "chunk_id": 0}],
        embeddings=[[0.0, 1.0]],
        source_name="new.pdf",
    )
    results = store.search([0.0, 1.0], top_k=5)
    assert len(results) == 1
    assert results[0]["source"] == "new.pdf"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _make_pipeline(tmp_path, llm=None):
    return RagPipeline(
        embedding_model=FakeEmbeddingModel(),
        vector_store=VectorStore(persist_dir=str(tmp_path)),
        llm=llm or FakeLLM(),
    )


def test_pipeline_process_document_summary(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    info = pipeline.process_document(make_sample_pdf(), source_name="pet_guide.pdf")
    assert info["file_name"] == "pet_guide.pdf"
    assert info["pages"] == 4
    assert info["char_count"] > 0
    assert info["chunk_count"] >= 4


def test_pipeline_ask_returns_answer_sources_and_chunks(tmp_path):
    llm = FakeLLM(answer="Cats need protein-rich food.")
    pipeline = _make_pipeline(tmp_path, llm=llm)
    pipeline.process_document(make_sample_pdf(), source_name="pet_guide.pdf")

    result = pipeline.ask("What should cats eat?", top_k=5)
    assert result["answer"] == "Cats need protein-rich food."
    assert 2 in result["sources"]                     # the feeding page
    assert result["chunks"][0]["page"] == 2           # most relevant first
    # The LLM prompt actually contained the retrieved context.
    assert "RETRIEVED DOCUMENT CONTEXT" in llm.last_user_prompt
    assert "cats" in llm.last_user_prompt.lower()


def test_pipeline_ask_empty_question_raises(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_pipeline_missing_api_key_raises_friendly_error():
    """Without GEMINI_API_KEY the client must fail with the documented message."""
    client = GeminiClient(api_key="")
    assert client.is_configured() is False
    with pytest.raises(LLMError) as exc_info:
        client.generate_answer("system", "user")
    assert MISSING_KEY_MESSAGE in str(exc_info.value)


def test_build_user_prompt_contains_context_and_question():
    chunks = [{"text": "cats eat fish", "page": 2, "source": "x.pdf", "similarity": 0.9}]
    prompt = build_user_prompt("What do cats eat?", chunks)
    assert "cats eat fish" in prompt
    assert "What do cats eat?" in prompt
    assert "Page 2" in prompt
