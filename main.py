import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
st.set_page_config(page_title="Ask your PDF")
st.header("Ask your PDF 😎")

# upload file
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    st.write(text)