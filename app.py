import os
import streamlit as st
import tempfile

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="My Local AI Agent", page_icon="🤖")
st.title("🤖 Chat with ANY PDF (Running Locally)")

# --- 2. SIDEBAR FOR API KEY ---
with st.sidebar:
    st.header("⚙️ Settings")
    # We let you paste the key in the browser so it's safe
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    
    # Upload File
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# --- 3. MAIN LOGIC ---
if groq_api_key and uploaded_file:
    # Set the key
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Save uploaded file temporarily so the Agent can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    st.success("✅ File Uploaded! AI is ready.")

    # --- CHAT INTERFACE ---
    user_query = st.text_input("Ask a question about your PDF:")

    if user_query:
        with st.spinner("Thinking... (This might take 5 seconds)"):
            try:
                # --- LAZY IMPORT (Only loads when needed) ---
                from langchain_groq import ChatGroq
                from langchain_community.document_loaders import PyPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain_chroma import Chroma
                from langchain_core.tools import tool
                from langgraph.prebuilt import create_react_agent

                # 1. Load & Split
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # 2. Embed (This uses the model you just downloaded)
                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
                retriever = vectorstore.as_retriever()

                # 3. Tool
                @tool
                def search_pdf(query: str) -> str:
                    """Search the PDF for answers."""
                    docs = retriever.invoke(query)
                    return "\n\n".join([d.page_content for d in docs])

                # 4. Agent
                llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
                agent = create_react_agent(llm, [search_pdf])

                # 5. Run
                response = agent.invoke({"messages": [("user", user_query)]})
                
                # Show Answer
                st.markdown(f"### 🤖 AI Answer:")
                st.write(response['messages'][-1].content)

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("👈 Please paste your API Key and upload a PDF to start.")