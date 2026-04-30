import streamlit as st
import os
import chromadb
import uuid
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# --- PAGE SETUP ---
st.set_page_config(page_title="My AI Agent", page_icon="🤖")
st.title(" Chat with PDF")

# --- INITIALIZE UPLOADER KEY ---
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = str(uuid.uuid4())

# --- GET SECRET API KEY ---
# This pulls YOUR hidden key from Streamlit's secure vault
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Configuration Error: The secret GROQ_API_KEY is missing. Please add it to your Streamlit Cloud Secrets.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    # Notice we completely removed the API key input box!
    
    uploaded_file = st.file_uploader(
        "Upload PDF", 
        type="pdf", 
        key=st.session_state["file_uploader_key"]
    )
    
    # --- RESET BUTTON LOGIC ---
    st.markdown("---") 
    if st.button("🔄 Complete Reset / Clear Data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["file_uploader_key"] = str(uuid.uuid4())
        st.rerun()

# --- MAIN LOGIC ---
# Now it only waits for an uploaded file
if uploaded_file:
    if "vectors" not in st.session_state:
        st.info("⏳ Processing PDF... please wait.")
        
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        unique_collection_name = str(uuid.uuid4())

        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name=unique_collection_name
        )
        
        st.session_state.vectors = vector_store
        st.success("PDF Processed! You can ask questions now.")
        st.rerun()

# --- CHAT INTERFACE ---
if "vectors" in st.session_state and st.session_state.vectors is not None:
    user_question = st.text_input("Ask a question about the PDF:")
    
    if user_question:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectors.as_retriever()
        )
        
        response = qa_chain.invoke(user_question)
        st.write(response["result"])
else:
    st.info("Please upload a PDF and wait for it to process before chatting.")