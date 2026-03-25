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
st.title("🤖 Chat with PDF")

# --- INITIALIZE UPLOADER KEY ---
# This is the magic trick to force the file uploader to clear itself
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = str(uuid.uuid4())

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    # Link to get the key if you forgot it
    st.markdown("[Get your Groq API Key](https://console.groq.com/keys)")
    api_key = st.text_input("Groq API Key", type="password")
    
    # We pass the dynamic key to the uploader here
    uploaded_file = st.file_uploader(
        "Upload PDF", 
        type="pdf", 
        key=st.session_state["file_uploader_key"]
    )
    
    # --- RESET BUTTON LOGIC ---
    st.markdown("---") 
    if st.button("🔄 Complete Reset / Clear Data"):
        # 1. Wipe everything stored in Streamlit's memory
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # 2. Generate a brand NEW key for the uploader so it visually clears out
        st.session_state["file_uploader_key"] = str(uuid.uuid4())
        
        # 3. Force the app to refresh instantly
        st.rerun()

# --- MAIN LOGIC ---
if api_key and uploaded_file:
    os.environ["GROQ_API_KEY"] = api_key
    
    # Check if we have already processed this specific file
    if "vectors" not in st.session_state:
        st.info("⏳ Processing PDF... please wait.")
        
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Load and Split the PDF
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Create the AI Brain (Embeddings)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Clean the system cache to prevent ChromaDB crashes
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        # Generate a totally unique name for the database to prevent data mixing
        unique_collection_name = str(uuid.uuid4())

        # Store in Database (ChromaDB) using the unique name
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name=unique_collection_name
        )
        
        # Save to session so we don't reload it every time
        st.session_state.vectors = vector_store
        st.success("PDF Processed! You can ask questions now.")
        st.rerun()

# --- CHAT INTERFACE ---
# Only show the chat box IF the vectors exist in the session state
if "vectors" in st.session_state and st.session_state.vectors is not None:
    user_question = st.text_input("Ask a question about the PDF:")
    
    if user_question:
        # 1. Use the new working model
        llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        
        # 2. Setup the Q&A Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectors.as_retriever()
        )
        
        # 3. Get the answer
        response = qa_chain.invoke(user_question)
        st.write(response["result"])
else:
    # If there are no vectors, tell them to upload
    st.info("Please upload a PDF and wait for it to process before chatting.")