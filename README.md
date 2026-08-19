# 📄 Intelligent Document Q&A Assistant (RAG)

**Ask questions about your PDF using Retrieval-Augmented Generation (RAG)**

An academic AI project: upload a PDF, and the app indexes it into a vector
database, then answers your questions **strictly from the document's
content** using Google's Gemini API — with sources shown for every answer.

---

## 1. Project Overview

Large language models (LLMs) are powerful, but they hallucinate and their
knowledge is frozen at training time. This project builds a complete
**Retrieval-Augmented Generation (RAG)** pipeline that solves both problems:
before the model answers, the app *retrieves* the most relevant passages of
*your* document and gives them to the model as context. The result is an
answer grounded in the document, with visible sources.

The application is built with **Streamlit**, processes PDFs with
**PyPDF**, chunks text with a simple custom splitter, embeds chunks locally
with **sentence-transformers**, stores vectors in **ChromaDB**, and
generates answers with Google's **Gemini** API.

## 2. Problem Statement

- Users need accurate answers about the *contents of their own documents*
  (manuals, research papers, lecture notes, reports).
- General-purpose chatbots cannot see these private documents and may
  fabricate answers.
- There is a need for a simple, verifiable Q&A system over uploaded PDFs.

## 3. Objectives

1. Accept a PDF upload and extract its text page by page.
2. Chunk the text and embed each chunk into a vector space.
3. Store the vectors in a persistent local vector database (ChromaDB).
4. Retrieve the most relevant chunks for a user's question (semantic search).
5. Generate an answer with Gemini using **only** the retrieved context.
6. Display the answer together with sources (page numbers + retrieved chunks).
7. Handle errors gracefully and keep the code simple enough to explain.

## 4. Features

- 📎 **PDF upload** (PDF-only, validated)
- 🔍 **Semantic search** over the document (cosine similarity)
- 💬 **Chat interface** with session-based history
- 📑 **Sources** for every answer + expandable retrieved context
- ⚙️ **Configurable** number of retrieved chunks (K = 3–8)
- 🧹 **Friendly error handling** (missing key, invalid PDF, scanned PDF,
  empty question, API failures, ...)
- 🧪 **Offline unit tests** (no API key required)
- 🐳 **Docker** support

## 5. RAG Architecture

```
PDF Upload → Text Extraction (pypdf) → Cleaning → Chunking
→ Embeddings (all-MiniLM-L6-v2) → ChromaDB Vector Storage
→ User Question → Question Embedding → Similarity Search
→ Top K Chunks → Prompt Construction → Gemini LLM
→ Answer + Sources
```

Two phases:

- **Ingestion (offline):** process the PDF, embed the chunks, store them.
- **Question answering (online):** embed the question, retrieve top-K
  chunks, build a grounded prompt, generate the answer, show sources.

## 6. Technology Stack

| Layer              | Technology                              |
|--------------------|-----------------------------------------|
| UI                 | Streamlit                               |
| PDF processing     | PyPDF (`pypdf`)                         |
| Text chunking      | Custom recursive splitter (simple)      |
| Embeddings         | `sentence-transformers` / `all-MiniLM-L6-v2` (384-dim, local) |
| Vector database    | ChromaDB (persistent, local)            |
| LLM                | Google Gemini (`google-genai` SDK)      |
| Config / secrets   | `python-dotenv` + `.env`                |
| Language           | Python 3.10+                            |

## 7. Project Structure

```
RAG-model/
│
├── app.py                     # Streamlit UI (entry point)
├── src/
│   ├── __init__.py
│   ├── config.py              # All settings from env vars
│   ├── pdf_processor.py       # PDF extraction, cleaning, chunking
│   ├── embeddings.py          # SentenceTransformer wrapper
│   ├── vector_store.py        # ChromaDB operations
│   ├── llm.py                 # Gemini client + friendly errors
│   └── rag_pipeline.py        # Retrieval + prompt + answer generation
├── data/
│   ├── .gitkeep
│   └── sample_document.pdf    # Sample PDF for the demo
├── chroma_db/                 # Vector DB storage (gitignored)
├── reports/
│   ├── RAG_Research_Report.md         # Part 1 report
│   └── Problem_Solving_Answers.md     # Part 3 answers
├── screenshots/README.md      # Which screenshots to take
├── tests/
│   ├── __init__.py
│   ├── utils.py               # PDF generator + fake models
│   ├── test_pdf_processor.py  # 10 tests
│   └── test_rag_components.py # 11 tests
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
└── demo_script.md             # 3–5 minute demo script
```

## 8. Installation

Requirements: **Python 3.10+** and **pip**.

```bash
# 1. Clone or copy the project, then enter the folder
cd RAG-model

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> ⚠️ **Note for the original development Mac:** this project folder was
> originally inside an iCloud-synced Desktop, and iCloud kept corrupting the
> venv and evicting source files. On that machine the project was moved to
> `~/RAG-model` (outside iCloud) and a pre-built venv exists at
> `~/rag-model-venv`. On any normal computer the standard steps above work
> perfectly.

## 9. API Key Configuration

1. Get a free Gemini API key: <https://aistudio.google.com/apikey>
2. Copy the template and fill it in:

```bash
cp .env.example .env
```

```ini
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.7-flash
```

- The key is **never hardcoded** — it is read from the environment.
- `.env` is in `.gitignore`, so it is never committed.
- `GEMINI_MODEL` is configurable; `gemini-3.7-flash` is the current stable
  Flash model (2.5 Flash still works but is being deprecated).

## 10. How to Run Locally

```bash
streamlit run app.py
```

Open **http://localhost:8501**, then:

1. Upload `data/sample_document.pdf` (or any text PDF).
2. Click **⚙️ Process Document**.
3. Ask a question, e.g. *"What should cats eat?"*
4. Read the answer + **Sources** + "View retrieved context".

## 11. How the RAG Pipeline Works

1. **Upload** → the PDF bytes are read.
2. **Extract** → `pypdf` reads text page by page; whitespace is cleaned.
3. **Chunk** → text is split into ~500-character chunks with overlap; each
   chunk remembers its page number.
4. **Embed** → `all-MiniLM-L6-v2` converts every chunk into a
   384-dimensional vector (semantically similar texts → similar vectors).
5. **Store** → vectors + metadata (source, page, chunk id) go into a fresh
   ChromaDB collection (old documents never mix with new ones).
6. **Ask** → the question is embedded with the same model.
7. **Retrieve** → ChromaDB returns the K most similar chunks.
8. **Prompt** → a system instruction ("answer only from the context, never
   invent facts") + the retrieved chunks + the question are combined.
9. **Generate** → Gemini answers with `temperature=0.2` (factual).
10. **Show** → answer + page sources + expandable retrieved context.

## 12. Example Usage

```
User:      What should cats eat?
Assistant: According to the document (Page 2), cats need protein-rich
           food to stay healthy. Provide fresh water every day, and feed
           adult cats twice a day following the portions on the package.
Sources:   Page 2
           [View retrieved context ▸]
```

If the document does not contain the answer, the app responds:
"I couldn't find this information in the uploaded document."

## 13. Screenshots

See [`screenshots/README.md`](screenshots/README.md) for the exact list.
Save your own screenshots into the `screenshots/` folder.

## 14. Error Handling

| Situation                          | Behaviour                                       |
|------------------------------------|-------------------------------------------------|
| Missing Gemini API key             | Friendly message with setup instructions        |
| Invalid / corrupted PDF            | Friendly "not a valid PDF" message              |
| Empty PDF                          | Friendly "empty file" message                   |
| Scanned / image-only PDF           | "No text could be extracted ... OCR not supported" |
| Extraction failure                 | Friendly message; details logged                |
| Gemini API / rate limit / quota    | Friendly retry message; details logged          |
| Embedding model load failure       | Friendly message with troubleshooting hints      |
| ChromaDB failure                   | Friendly message                                |
| Empty question                     | Warning "Please type a question first."         |
| Question before upload/processing  | Chat input is disabled                          |

Errors are logged (for developers) and shown as friendly `st.error` messages
to users — no raw tracebacks.

## 15. Limitations

- **Scanned PDFs** need OCR (not included).
- Retrieval quality depends on the chunking and the embedding model.
- Requires an internet connection + Gemini API key for answers.
- Embedding model downloads on first use (a few tens of MB).

## 16. Future Improvements

- OCR support for scanned PDFs (e.g. Tesseract).
- Hybrid search (BM25 + embeddings) and re-ranking.
- Multiple document uploads with per-document collections.
- Chat history persisted to disk (e.g. SQLite) instead of session-only.
- Model selection dropdown (Gemini models, temperature).

## 17. Deployment Instructions

**Option A — Docker (local or any server):**

```bash
docker build -t rag-qa .
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key_here rag-qa
# open http://localhost:8501
```

**Option B — Streamlit Community Cloud (free):**

1. Push the project to GitHub (add your `.env` values as **Secrets** in the
   Streamlit Cloud dashboard instead of committing them).
2. Import the repository at <https://streamlit.io/cloud>.
3. Set the secret `GEMINI_API_KEY` in **Settings → Secrets**.
4. Deploy — the app is served automatically.

> TODO: Student must complete this step — add your live deployment URL here:
> `Live deployment URL: [Add your deployment link here]`

## 18. GitHub Instructions

```bash
git init
git add .
git commit -m "AI Document Q&A (RAG) — Intelligent Document Q&A Assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> GitHub Repository: [Add your repository link here]
>
> TODO: Student must complete this step.

## 19. Demo Instructions

See [`demo_script.md`](demo_script.md) — a 3–5 minute spoken script covering
introduction, the problem, the stack, upload, processing, asking, sources,
chat history, the RAG architecture, error handling, and conclusion.

## 20. Testing

Run the unit tests (fully offline — **no API key needed**):

```bash
python -m pytest tests/ -v
```

All **21 tests pass**. They cover PDF extraction, empty/invalid/scanned PDF
handling, chunk creation, overlap, embedding generation, vector store
creation/search, collection isolation, prompt building, and the full
pipeline with mocked LLM calls.

## 21. Author / Student

| Field      | Value |
|------------|-------|
| Name       | TODO: Student must complete this step. |
| College    | TODO: Student must complete this step. |
| Roll No.   | TODO: Student must complete this step. |
| Course     | TODO: Student must complete this step. |
| Submission | TODO: Student must complete this step. |

---

Built with Streamlit, PyPDF, sentence-transformers, ChromaDB, and Google
Gemini. Part of an academic assignment on Retrieval-Augmented Generation.
