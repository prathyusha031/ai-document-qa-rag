# Screenshots

Take these screenshots for your report/submission and save them in this
folder. The `data/sample_document.pdf` file is provided so the app can be
demoed without searching for a PDF.

| # | What to capture                                                       | How                                              |
|---|-----------------------------------------------------------------------|--------------------------------------------------|
| 1 | Home screen (before upload)                                           | Run the app and screenshot the main area         |
| 2 | Sidebar with Gemini API status (green = key configured)               | After adding your key to `.env`                  |
| 3 | Uploading the PDF (`data/sample_document.pdf`)                        | Click "Browse files" in the sidebar              |
| 4 | "Document processed" summary (file, pages, characters, chunks)        | Click **Process Document** and wait              |
| 5 | Asking a question (e.g. "What should cats eat?")                      | Type in the chat box and press Enter             |
| 6 | Generated answer with **Sources: Page 2**                             | The answer appears in the chat                   |
| 7 | "View retrieved context" expander open (shows the chunks)             | Click the expander under the answer              |
| 8 | Chat history (ask 2–3 questions)                                      | Shows previous messages are kept                 |
| 9 | Clear Chat button working                                             | Click it and screenshot the empty chat           |
| 10 | Missing API key warning (if demoing without a key)                    | Remove `.env` temporarily                        |
| 11 | Terminal showing the tests passing (`pytest -v`)                      | Run `python -m pytest tests/ -v`                 |
| 12 | Terminal showing `streamlit run app.py` + Local URL                   | Start the app                                    |

> NOTE: Do not fabricate screenshots. Take them from your own run.
