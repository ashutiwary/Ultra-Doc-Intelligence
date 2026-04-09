# Ultra Doc-Intelligence

A simple AI chatbot that lets you upload a logistics document (PDF, DOCX, or TXT) and ask questions about it in plain English. It also has a one-click button to pull out key fields as structured JSON.

## Tech Stack

- **Frontend:** Streamlit (single-file app, no separate backend needed)
- **Document Parsing:** pypdf (PDF), python-docx (DOCX), plain text for TXT files
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API needed)
- **Vector Database:** ChromaDB (in-memory, scoped per session)
- **LLM:** Groq API with Llama 3.3 70B

## How it works

1. Upload a document through the sidebar.
2. The app extracts the text, splits it into chunks, and stores the embeddings in an in-memory ChromaDB instance.
3. When you ask a question, the app finds the most relevant chunks and sends them to the LLM along with your question.
4. The LLM answers based only on what is in the document.

Each new upload clears the previous document, so the chat is always scoped to whatever you last uploaded.

## Features

- Chat with any logistics document
- Confidence score on every answer
- Guardrail that returns "Not found in document" if the answer is not in the document
- One-click structured JSON extraction of key logistics fields (Shipment ID, shipper, consignee, rate, stops)

## Guardrails

If the best matching chunk has a vector distance above 1.5, the system returns "Not found in document" instead of guessing. This prevents the LLM from making up answers.

## How to run locally

**1. Clone the repo**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up your API key**

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Or create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

**5. Run the app**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud (free)

1. Push your code to a public GitHub repo (make sure `.streamlit/secrets.toml` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with Google
3. Connect your GitHub account and select your repo
4. In "Advanced settings", add your secret:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click Deploy

## Known Limitations

- Complex tables that span multiple pages may not parse well since the row and header relationship can get broken during text extraction.
- The Llama model on Groq occasionally misses optional fields in the JSON extraction compared to larger proprietary models.
- The vector store is in-memory, so it resets if the app restarts or the session ends.

## Ideas for improvement

- Add a reranker (like Cohere) to improve which chunks get sent to the LLM.
- Use the `Instructor` library to enforce strict Pydantic schema validation on the extraction output.
- Add persistent storage so the document survives page refreshes.
