import streamlit as st
import requests

# FastAPI backend URL
backend_url = "http://127.0.0.1:8000/summarize"

st.title("PDF Summarizer")

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    st.info("File uploaded successfully!")

    # Send the PDF to the FastAPI backend
    if st.button("Generate Summary"):
        with st.spinner("Processing..."):
            files = {"file": uploaded_file}
            response = requests.post(backend_url, files=files)
            
            if response.status_code == 200:
                summary = response.json().get("summary", "No summary available.")
                st.success("Summary generated!")
                st.write(summary)
            else:
                st.error("Failed to generate summary. Please check the backend.")
