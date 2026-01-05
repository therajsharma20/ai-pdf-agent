import streamlit as st
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="My AI Agent", page_icon="🤖")
st.title("🤖 Chat with PDF")

# --- IMPORT CHECK (To prove libraries are working) ---
try:
    from langchain_groq import ChatGroq
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain.chains import RetrievalQA
    print("✅ All libraries imported successfully!")
except ImportError as e:
    st.error(f"❌ Library Error: {e}")
    st.error("Did you run the 'pip install' command in cmd?")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    # Link to get the key if you forgot it
    st.markdown("[Get your Groq API Key](https://console.groq.com/keys)")
    api_key = st.text_input("Groq API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

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
        
        # Store in Database (ChromaDB)
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # Save to session so we don't reload it every time
        st.session_state.vectors = vector_store
        st.success("PDF Processed! You can ask questions now.")
        st.rerun()

# --- CHAT INTERFACE ---
if "vectors" in st.session_state:
    user_question = st.text_input("Ask a question about the PDF:")
    
    if user_question:
        # Use the NEW working model
        llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        
        # Setup the Q&A Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectors.as_retriever()
        )
        
        # Get the answer
        response = qa_chain.invoke(user_question)
        st.write(response["result"])