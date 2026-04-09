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
    if vs is None:
        return {"answer": "No document loaded.", "confidence": 0.0}

    results = vs.similarity_search_with_score(question, k=3)

    if not results or results[0][1] > 1.5:
        return {"answer": "Not found in document.", "confidence": 0.0}

    context = "\n\n".join([doc.page_content for doc, score in results])
    prompt = f"Answer only from the context below.\nContext: {context}\nQuestion: {question}"

    response = load_llm().invoke(prompt)
    confidence = max(0.0, 1.0 - (results[0][1] / 2.0))

    return {
        "answer": response.content,
        "confidence": round(confidence, 2),
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
    st.write("Upload a logistics document to start chatting.")

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
