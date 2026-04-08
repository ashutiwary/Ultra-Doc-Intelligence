import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("Logistics Document AI")

st.header("1. Upload Document")
uploaded_file = st.file_uploader("Upload Logistics Document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Parsing and indexing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                st.success(f"Success: Indexed {response.json()['chunks_created']} chunks.")
            else:
                st.error("Failed to process document.")

st.divider()

st.header("2. Ask Questions")
question = st.text_input("Enter your question about the document:")
if st.button("Ask"):
    with st.spinner("Retrieving answer..."):
        response = requests.post(f"{API_URL}/ask", json={"question": question})
        if response.status_code == 200:
            data = response.json()
            st.write(f"**Answer:** {data['answer']}")
            st.write(f"**Confidence Score:** {data['confidence_score']}")
            
            with st.expander("View Grounding Sources"):
                for i, source in enumerate(data['sources']):
                    st.text_area(f"Source {i+1}", source, height=150)

st.divider()

st.header("3. Structured Extraction")
if st.button("Extract Shipment Data"):
    with st.spinner("Extracting JSON data..."):
        response = requests.post(f"{API_URL}/extract")
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error("Failed to extract data.")