# Retrieval-Augmented Generation (RAG): Architecture, Applications, Advantages, Limitations and Future Scope

---

## 1. Introduction

Artificial Intelligence (AI) aims to build systems that can perform tasks
that normally require human intelligence, such as understanding language,
recognising images, and making decisions. A major breakthrough in recent
years is the Large Language Model (LLM) — a deep neural network trained on
enormous amounts of text that can generate human-like responses, translate
languages, write code, and answer questions.

However, LLMs have two well-known problems. First, their knowledge is fixed
at training time: they do not know about events or documents that appeared
after their training data was collected. Second, they *hallucinate* — they
confidently generate plausible-sounding but incorrect information, because
they are simply predicting the most likely next words, not consulting a
source of truth. These limitations make LLMs risky for tasks that require
accurate, up-to-date answers about specific documents (for example, a
company manual, a legal contract, or a research paper).

**Retrieval-Augmented Generation (RAG)** was introduced to solve these
problems. Instead of asking the model to answer from memory alone, RAG
first *retrieves* relevant information from an external knowledge source
and then *generates* an answer grounded in that retrieved information.

## 2. What is RAG?

RAG is a framework that combines two components:

1. **A retriever** — a system that searches a large collection of documents
   (stored as vector embeddings in a vector database) and returns the most
   relevant passages for a user's question.
2. **A generator** — an LLM that produces the final answer using the
   retrieved passages as context.

The key idea is simple: *give the model the right information before asking
it to answer.* Because the model's answer is grounded in retrieved evidence,
it is more accurate, traceable, and up to date than a purely parametric
("memorised") response. The concept was formalised in the paper
"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
(Lewis et al., 2020).

## 3. Working Principle

The RAG pipeline (as implemented in this project) works in two phases:

**Phase A — Document ingestion (offline):**

1. **Document ingestion:** the user uploads a PDF.
2. **Text extraction:** the text is read page by page (using `pypdf`).
3. **Text cleaning:** extra whitespace is removed so chunks are tidy.
4. **Chunking:** the text is split into small, meaningful pieces of ~500
   characters with overlap (custom recursive splitter).
5. **Embedding generation:** each chunk is converted into a numerical vector
   using a sentence-transformer model (`all-MiniLM-L6-v2`, 384 dimensions).
   Chunks that are semantically similar end up close together in vector space.
6. **Vector storage:** the chunk vectors, together with metadata (source
   file name, page number, chunk ID), are stored in a vector database
   (ChromaDB).

**Phase B — Question answering (online):**

7. **Question embedding:** the user's question is embedded with the same model.
8. **Similarity search:** the vector database finds the K chunks whose
   vectors are closest (cosine similarity) to the question vector.
9. **Prompt construction:** a prompt is built from a system instruction
   ("answer only from the context, do not invent facts"), the retrieved
   chunks, and the user's question.
10. **LLM generation:** the LLM (Google Gemini) produces the answer grounded
    in the retrieved context.
11. **Answer + sources:** the answer is displayed together with the page
    numbers and an expandable view of the retrieved chunks.

## 4. RAG Architecture

```
                         ┌───────────────────────┐
                         │   User uploads a PDF  │
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │  Text extraction      │  (pypdf)
                         │  + cleaning           │
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │  Text chunking        │  (recursive splitter)
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │  Embeddings           │  (all-MiniLM-L6-v2)
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │  Vector database      │  (ChromaDB)
                         └───────────┬───────────┘
                                     │
              User question ─────────┤
                                     ▼
                         ┌───────────────────────┐
                         │  Similarity search    │  (top K chunks)
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │  Prompt construction  │  (context + question)
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │  LLM generation       │  (Gemini)
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │  Answer + sources     │  (Streamlit UI)
                         └───────────────────────┘
```

## 5. Real-World Applications

- **Enterprise knowledge assistants:** employees ask questions about
  internal manuals, policies, and product documentation.
- **Customer support:** support bots answer from FAQs and product manuals,
  reducing hallucinated answers and escalations.
- **Education:** students query textbooks and lecture notes; answers cite
  the exact pages (as in this project).
- **Legal document search:** lawyers retrieve relevant clauses from
  contracts and case documents.
- **Research:** researchers search and summarise large collections of
  academic papers.
- **Healthcare information retrieval:** clinicians can retrieve relevant
  medical literature (RAG itself does not make diagnoses — it only finds
  and summarises documents, and medical claims must always be verified).
- **Internal company documentation:** new employees ask questions about
  onboarding and internal systems.

## 6. Advantages

1. **Reduced hallucinations:** answers are grounded in retrieved evidence
   instead of the model's memory.
2. **Up-to-date knowledge:** new documents can be added without re-training
   the model.
3. **Traceable answers:** sources (page numbers, chunks) can be shown, so
   answers can be verified.
4. **No fine-tuning needed:** RAG works with a general-purpose LLM; the
   knowledge comes from the retrieval index.
5. **Domain flexibility:** the same model can serve different domains simply
   by pointing it at different document collections.
6. **Lower cost than training:** storing embeddings and retrieving is much
   cheaper than fine-tuning or training a model.

## 7. Limitations

1. **Retrieval quality is critical:** if the retriever returns irrelevant
   chunks, the answer will be wrong ("garbage in, garbage out").
2. **Chunking is a design choice:** badly sized chunks can lose context or
   split important information across chunks.
3. **Scanned/image PDFs need OCR:** documents without a text layer cannot be
   processed without an OCR engine (this project detects and reports this).
4. **Still not perfect:** even with good retrieval, the LLM can occasionally
   ignore the context or misquote it.
5. **Latency and complexity:** a RAG system has more moving parts (embedding
   model, vector DB, LLM) than a plain chat model.
6. **Dependency on an external LLM API:** quality, cost, and availability
   depend on the API provider.

## 8. Future Scope

- **Multimodal RAG:** retrieving and understanding images, tables, audio,
  and video alongside text.
- **Agentic RAG:** agents that decide *what* to retrieve, *when* to retrieve
  again, and how to combine multiple sources.
- **Better retrieval:** dense + sparse hybrid search (e.g. BM25 combined
  with embeddings) and re-ranking models.
- **Graph RAG:** using knowledge graphs to capture relationships between
  entities for more structured answers.
- **Improved evaluation:** standard benchmarks and metrics for RAG quality
  (retrieval accuracy and answer faithfulness).
- **Domain-specific RAG:** fine-tuned embedders and retrievers for fields
  like medicine, law, and finance.

## 9. Conclusion

RAG combines the strengths of information retrieval and generative AI: it
makes LLMs accurate, verifiable, and up to date while avoiding expensive
re-training. This project demonstrates a complete RAG application — a PDF
question-answering assistant in which the model answers strictly from the
uploaded document and shows its sources. While RAG has limitations, it is
currently one of the most practical and widely adopted ways to deploy LLMs
in real, knowledge-intensive applications, and its future (multimodal,
agentic, hybrid) promises even broader impact.

## 10. References

1. Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented
   Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401.
   https://arxiv.org/abs/2005.11401
2. Gao, Y., Xiong, Y., Gao, X., et al. (2023). *Retrieval-Augmented
   Generation for Large Language Models: A Survey*. arXiv:2312.10997.
   https://arxiv.org/abs/2312.10997
3. Google. *Gemini API documentation*.
   https://ai.google.dev/gemini-api/docs
4. Chroma. *ChromaDB documentation*. https://docs.trychroma.com/
5. Sentence-Transformers. *all-MiniLM-L6-v2 model card*.
   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
6. Streamlit. *Streamlit documentation*. https://docs.streamlit.io/
7. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings
   using Siamese BERT-Networks*. arXiv:1908.10084.
   https://arxiv.org/abs/1908.10084
