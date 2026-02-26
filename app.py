
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("📄 PDF Chatbot")

# Load Embedding Model
embeddings = HuggingFaceEmbeddings()

# Load Fixed PDF
pdf_path = "myfile.pdf"  # 🔥 Change this to your PDF name

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# Create or Load FAISS
if os.path.exists("vectorstore/index.faiss"):
    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore")


st.success("✅ Vectorstore Ready!")
