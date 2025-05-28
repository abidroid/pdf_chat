# import streamlit as st
# from dotenv import load_dotenv
# from pypdf import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.chat_models import ChatOpenAI
# from langchain.callbacks import get_openai_callback
# import os

# load_dotenv()
# st.set_page_config(page_title="Ask your PDF")
# st.header("Ask your PDF 😎")

# # upload file
# pdf = st.file_uploader("Upload your PDF", type="pdf")

# if pdf is not None:

#     # Define FAISS index path
#     index_path = "faiss_index"

#     # Check if the FAISS index already exists
#     if os.path.exists(index_path):
#         st.write("Loading existing embeddings...")
#         knowledge_base = FAISS.load_local(index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#     else:
#         st.write("Generating embeddings...")
        
#         # Extract text from PDF
#         pdf_reader = PdfReader(pdf)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()

#                 # split into chunks
#         text_splitter = CharacterTextSplitter(
#             separator="\n",
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text)

#         # Create embeddings and FAISS index
#         embeddings = OpenAIEmbeddings()
#         knowledge_base = FAISS.from_texts(chunks, embeddings)

#         # Save FAISS index to disk
#         knowledge_base.save_local(index_path)
#         st.write("Embeddings generated and saved!")

#     # Show user input
#     user_question = st.text_input("Ask a question about your PDF:")
#     if user_question:
#         docs = knowledge_base.similarity_search(user_question)

#         llm = ChatOpenAI(model="gpt-4")
#         chain = load_qa_chain(llm, chain_type="stuff")
#         with get_openai_callback() as cb:
#             response = chain.run(input_documents=docs, question=user_question)
#             print(cb)

#         st.write(response)

import streamlit as st
import hashlib
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

# Load environment variables from .env
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="Ask your PDF")
st.header("Ask your PDF 😎")

# File upload
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    # Generate a unique hash for each PDF to differentiate indexes
    pdf_bytes = pdf.read()
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
    index_path = f"faiss_index_{pdf_hash}"
    pdf.seek(0)  # Reset pointer after reading bytes

    if os.path.exists(index_path):
        st.write("🔁 Loading existing embeddings...")
        knowledge_base = FAISS.load_local(index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        st.write("⚙️ Generating embeddings from uploaded PDF...")

        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Split text into manageable chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create vector store using OpenAI embeddings + FAISS
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Save the index for future use
        knowledge_base.save_local(index_path)
        st.success("✅ Embeddings generated and saved!")

    # Chat input
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        # Perform similarity search
        docs = knowledge_base.similarity_search(user_question)

        # Use OpenAI LLM (ChatOpenAI)
        llm = ChatOpenAI(model="gpt-4")  # or "gpt-3.5-turbo" for cheaper/faster

        # Load the chain using deprecated interface (consider migration later)
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)
            print(cb)  # shows token usage in the console/log
