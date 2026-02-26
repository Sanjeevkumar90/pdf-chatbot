
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
pdf_path = "pharma_dictionary.pdf" # 🔥 Change this to your PDF name

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
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Load LLM (Free HF model)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Question Input Box
query = st.text_input("💬 Ask a question about the PDF:")

if query:
    with st.spinner("Thinking..."):
        result = qa.run(query)
        st.write("### 📌 Answer:")
        st.write(result)

