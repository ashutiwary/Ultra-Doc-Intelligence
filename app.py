import io
import os
import json
import warnings
import logging

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Support both local .env and Streamlit Cloud secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# --- Cached Resources (load once, shared across reruns) ---

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)

# --- Core Logic ---

def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs)
    else:
        return file_bytes.decode("utf-8")

def process_document(file_bytes: bytes, filename: str) -> int:
    text = extract_text(file_bytes, filename)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])

    st.session_state.vector_store = FAISS.from_documents(chunks, load_embeddings())
    st.session_state.doc_text = text
    return len(chunks)

def ask_question(question: str) -> dict:
    vs = st.session_state.get("vector_store")
    
    # --- SCENARIO A: No Document Uploaded ---
    if vs is None:
        # We let the LLM decide how to respond based purely on this prompt
        no_doc_prompt = (
            "You are a helpful AI Logistics Assistant. Currently, NO document has been uploaded by the user.\n"
            "Follow these rules based on the User Input:\n"
            "1. If it is a greeting or pleasantry, say hello and politely ask them to upload a document to get started.\n"
            "2. If it is a specific question, explain that you need them to upload a document first before you can answer.\n\n"
            f"User Input: {question}"
        )
        response = load_llm().invoke(no_doc_prompt)
        return {"answer": response.content, "confidence": 0.0}

    # --- SCENARIO B: Document IS Uploaded ---
    results = vs.similarity_search_with_score(question, k=4)

    if not results:
        return {"answer": "Could not find relevant content in the document.", "confidence": 0.0}

    context = "\n\n".join([doc.page_content for doc, score in results])
    
    # We let the LLM handle greetings vs. document retrieval here
    rag_prompt = (
        "You are a helpful AI logistics assistant analyzing a document. Follow these strict rules:\n"
        "1. If the User Input is a greeting or casual conversation, respond politely and ask what they want to know about their document.\n"
        "2. If the User Input is a question, answer it using ONLY the Context provided below.\n"
        "3. If the answer is not contained in the Context, say exactly: "
        "'I could not find that information in the uploaded document.'\n\n"
        f"Context:\n{context}\n\n"
        f"User Input: {question}"
    )

    response = load_llm().invoke(rag_prompt)
    best_score = results[0][1]
    confidence = max(0.0, round(1.0 - (best_score / 2.0), 2))

    return {
        "answer": response.content,
        "confidence": confidence,
    }

def extract_data() -> dict:
    full_text = st.session_state.get("doc_text", "")
    if not full_text:
        return {"error": "No document loaded."}

    prompt = (
        "You are a data extraction tool. Extract the following fields into a VALID JSON object: "
        "Shipment_id, shipper, consignee, rate, and stops. "
        "Do not include any extra text or markdown code blocks. "
        f"Text: {full_text[:6000]}"
    )

    response = load_llm().invoke(prompt)
    try:
        return json.loads(
            response.content.replace("```json", "").replace("```", "").strip()
        )
    except Exception:
        return {"error": "Failed to parse JSON", "raw": response.content}

# --- UI ---

st.set_page_config(page_title="Ultra Doc-Intelligence", layout="wide")

with st.sidebar:
    st.title("Document Center")

    st.info("**How to get started:**\n1. Upload a PDF, DOCX, or TXT file below.\n2. Click **Process Document** to index it.\n3. Ask questions in the chat on the right.")

    uploaded_file = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])

    if st.button("Process Document", use_container_width=True):
        if uploaded_file:
            with st.spinner("Processing document..."):
                try:
                    count = process_document(uploaded_file.getvalue(), uploaded_file.name)
                    st.success(f"Done. {count} segments indexed.")
                    st.session_state.doc_loaded = True
                    st.session_state.messages = []
                except Exception as e:
                    st.error(f"Failed to process document: {e}")
        else:
            st.warning("Please select a file first.")

    st.divider()

    st.caption("**Run Data Extraction** automatically pulls key logistics fields (Shipment ID, shipper, consignee, rate, stops) from the processed document and displays them as JSON.")

    if st.button("Run Data Extraction", use_container_width=True):
        with st.spinner("Extracting fields..."):
            try:
                st.json(extract_data())
            except Exception as e:
                st.error(f"Extraction failed: {e}")

# --- Chat Window ---

st.title("Ultra Doc-Intelligence")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "confidence" in message:
            st.caption(f"Confidence: {message['confidence']}")

if prompt := st.chat_input("Ask a question about your document..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                data = ask_question(prompt)
                st.markdown(data["answer"])
                st.caption(f"Confidence: {data['confidence']}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "confidence": data["confidence"],
                })
            except Exception as e:
                st.error(f"Something went wrong: {e}")
