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
st.set_page_config(page_title="Semantic Knowledge Weaver", page_icon="🤖", layout="centered")
st.title("🤖 Chat with your PDF")
st.markdown("Upload a document and instantly ask questions about its contents.")

# --- INITIALIZE UPLOADER KEY ---
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = str(uuid.uuid4())

# --- SECURE API KEY HANDLING ---
# This pulls YOUR hidden key from Streamlit's secure vault
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error(" Configuration Error: The secret GROQ_API_KEY is missing. Please add it to your Streamlit Cloud Secrets.")
    st.stop()

# --- SIDEBAR & SETTINGS ---
with st.sidebar:
    st.header("Upload Document")
    
    # The file uploader uses the dynamic key to allow complete visual resets
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf", 
        key=st.session_state["file_uploader_key"]
    )
    
    st.markdown("---") 
    
    # The Nuclear Reset Button
    if st.button("🔄 Start Fresh / Clear Data"):
        # 1. Wipe everything stored in Streamlit's memory
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # 2. Generate a brand NEW key for the uploader so it visually clears out
        st.session_state["file_uploader_key"] = str(uuid.uuid4())
        
        # 3. Force the app to refresh instantly
        st.rerun()

    st.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.caption("Powered by Meta Llama 3.3 & LangChain")

# --- MAIN RAG LOGIC ---
if uploaded_file:
    # Check if we have already processed this specific file
    if "vectors" not in st.session_state:
        with st.spinner(" Analyzing document and building knowledge base..."):
            
            # Save the uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Load the PDF
            loader = PyPDFLoader("temp.pdf")
            try:
                docs = loader.load()
            except Exception as e:
                st.error(f" Could not read the PDF file: {e}")
                st.stop()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # --- THE FAILSAFE: Check for empty/scanned PDFs ---
            if len(splits) == 0:
                st.error(" Error: No readable text found in this PDF. It might be a scanned image or an empty file. Please upload a PDF with readable text.")
                st.stop()
            
            # Create Embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Clean system cache to prevent ChromaDB crashes
            chromadb.api.client.SharedSystemClient.clear_system_cache()

            # Generate a unique database name to prevent data mixing (Contamination Fix)
            unique_collection_name = "collection_" + str(uuid.uuid4()).replace("-", "")

            # Store in Database
            try:
                vector_store = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings,
                    collection_name=unique_collection_name
                )
            except Exception as e:
                st.error(f"🚨 Failed to build Vector Database: {e}")
                st.stop()
            
            # Save to session state
            st.session_state.vectors = vector_store
            st.success(" Document successfully indexed! You can ask questions below.")

# --- CHAT INTERFACE ---
if "vectors" in st.session_state and st.session_state.vectors is not None:
    st.markdown("### Ask a Question")
    user_question = st.chat_input("E.g., What is the main conclusion of this report?")
    
    if user_question:
        # Display user message
        st.chat_message("user").write(user_question)
        
        # Display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Initialize LLM
                llm = ChatGroq(model_name="llama-3.3-70b-versatile")
                
                # Setup QA Chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectors.as_retriever()
                )
                
                # Get response
                try:
                    response = qa_chain.invoke(user_question)
                    st.write(response["result"])
                except Exception as e:
                    st.error(f" An error occurred while generating the answer: {e}")
else:
    if not uploaded_file:
        st.info("👆 Please upload a PDF in the sidebar to begin.")
