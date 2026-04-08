# Ultra Doc-Intelligence: Logistics AI Assistant

A Proof of Concept (POC) AI system that allows users to upload logistics documents (Rate Confirmations, BOLs), interact with them using natural language, and extract structured JSON data.

## Architecture & Tech Stack
* **Backend:** FastAPI (Python) for modular API endpoints.
* **Frontend:** Streamlit for a lightweight, interactive UI.
* **Document Parsing:** LlamaParse to convert dense logistics tables into Markdown.
* **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (Local, fast).
* **Vector Database:** ChromaDB (Local persistent storage).
* **LLM Orchestration:** LangChain / Groq API (Llama 3) for high-speed inference.

## API Endpoints
* `POST /upload`: Accepts PDF/DOCX/TXT files, parses text (preserving tables), chunks data, and stores embeddings in ChromaDB.
* `POST /ask`: Accepts natural language questions, retrieves context via RAG, applies hallucination guardrails, and returns an answer with a confidence score and sources.
* `POST /extract`: Extracts a strict JSON schema containing: `Shipment_id, shipper, consignee, pickup_datetime, delivery_datetime, equipment_type, mode, rate, currency, weight, carrier_name`. Returns nulls for missing fields.

## Core Strategies

### Chunking Strategy
Logistics documents contain dense tables. Standard character splitting destroys table context. 
1. **Markdown Parse:** We use LlamaParse to extract tables as Markdown.
2. **Semantic Split:** `MarkdownHeaderTextSplitter` groups related key-value pairs.
3. **Token Limit Split:** `RecursiveCharacterTextSplitter` (chunk_size=500) ensures chunks fit within the strict 256-token limit of the MiniLM embedding model.

### Retrieval Method
The system embeds the user query and performs a similarity search (`k=3`) against the ChromaDB vector index to retrieve the most contextually relevant document chunks.

### Guardrails Approach
We implemented a strict threshold guardrail to prevent hallucinations. If the vector distance (L2 metric) of the best matching chunk exceeds a threshold of `1.5`, the system refuses to answer and returns "Not found in document".

### Confidence Scoring Method
The confidence score is a heuristic calculation derived from the L2 distance of the top retrieved chunk. It normalizes the distance into a 0.0 to 1.0 scale, where a lower distance yields a higher confidence percentage.

## Known Failure Cases
* **Complex Multi-Page Tables:** If a table spans multiple pages, LlamaParse may occasionally break the header-to-row relationship.
* **Strict JSON Adherence:** Open-weight models on Groq are fast but can sometimes miss optional fields in the structured extraction schema compared to proprietary models.

## Improvement Ideas
* Integrate a reranker (like Cohere) to improve retrieval accuracy before sending context to the LLM.
* Use the `Instructor` Python library to enforce stricter Pydantic schema validation for the structured extraction endpoint.

## How to Run Locally

1. Setup Environment:
    python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Environment Variables:
    export GROQ_API_KEY="your_key"
export LLAMA_CLOUD_API_KEY="your_key"

3. Run Backend:
    uvicorn main:app --reload

4. Run Frontend:
    source venv/bin/activate
streamlit run app.py