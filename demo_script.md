# Demo Script (3–5 minutes)

> Replace [Your Name] / [College] with your details. Speak naturally — this
> script is a guide, not something to read word for word.

---

**[0:00 — Introduction]**

> "Good morning, everyone. My name is [Your Name] from [College]. Today I'm
> presenting my project, **Intelligent Document Q&A Assistant** — an
> AI application that lets you upload a PDF and ask questions about its
> content in plain English, powered by Retrieval-Augmented Generation, or
> RAG."

**[0:20 — Problem being solved]**

> "Large language models like Gemini are very powerful, but they have two
> big problems: their knowledge is frozen at training time, and they can
> hallucinate — making up facts that sound convincing but are wrong. If you
> ask a chatbot about a document it has never seen, it will guess. My
> project solves this by combining a **retriever** with a **generator**:
> the app first finds the relevant parts of your document, and only then
> asks the model to answer — using that evidence."

**[0:40 — Technology stack]**

> "The stack is: **Streamlit** for the user interface, **PyPDF** for text
> extraction, a **custom recursive text splitter** for chunking,
> **sentence-transformers** with the `all-MiniLM-L6-v2` model for local
> embeddings, **ChromaDB** as the vector database, and Google's **Gemini**
> API for answer generation."

**[1:00 — Uploading and processing the PDF]**

> "Let me show you. I'll upload the sample document — a pet care guide.
> In the sidebar I click Process Document. Behind the scenes the app
> extracts the text page by page, cleans it, splits it into small chunks,
> converts each chunk into a 384-dimensional vector, and stores those
> vectors in ChromaDB. You can see the summary: 4 pages, 576 characters,
> 4 chunks."

**[1:40 — Asking a question]**

> "Now let me ask: *'What should cats eat?'* The question is converted to
> the same kind of vector, the vector database finds the chunks most
> similar to the question — the feeding page — and Gemini generates an
> answer using only that context."

**[2:10 — Showing the answer and sources]**

> "Notice the answer is grounded in the document, and below it we see
> **Sources: Page 2**. If I expand **View retrieved context**, we can see
> exactly which chunk was used. This proves the answer came from the
> document, not from the model's imagination."

**[2:40 — Chat history]**

> "I can keep asking questions and the conversation is maintained — the
> history stays in the session. If I click **Clear Chat**, the conversation
> resets."

**[3:00 — Explaining the RAG architecture]**

> "In one sentence: RAG = **retrieve** + **augment** + **generate**. We
> retrieve relevant passages from a vector database, augment the prompt
> with them, and generate an answer that is grounded in those passages.
> This reduces hallucinations, keeps knowledge up to date, and lets the
> app answer questions about *any* document without re-training the model."

**[3:30 — Error handling]**

> "The app also handles errors gracefully. For example, if I ask a question
> without a valid Gemini API key, the user sees a friendly message instead
> of a crash. It also detects invalid PDFs, empty PDFs, and scanned
> image-only PDFs, and tells the user what's wrong."

**[3:50 — Conclusion]**

> "To summarise: this project demonstrates a complete, modular RAG
> pipeline — PDF processing, chunking, embeddings, vector search, prompt
> engineering, and LLM generation — in a clean, beginner-friendly
> Streamlit app. Thank you. I'm happy to take questions."

---

## Tips

- Practice once with the sample PDF and the question *"What should cats eat?"*.
- If the live Gemini call is not available during the demo, the app still
  shows the retrieval step and a friendly missing-key message — use
  screenshot #10 to cover this.
- Keep the slides minimal; the live demo is the strongest part.
