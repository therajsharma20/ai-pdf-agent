import os
import sys
import warnings

# --- 1. SETUP & CLEANUP ---
warnings.filterwarnings("ignore") # Hide the scary yellow text
os.environ["GROQ_API_KEY"] = "ENTER_KEY_HERE" # <--- PASTE KEY HERE

# --- IMPORTS ---
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- 2. LOAD THE PDF (The "Eyes") ---
print("--- 1. READING PDF (This happens once) ---")
# We load the file from your Desktop
loader = PyPDFLoader("test.pdf")
docs = loader.load()

# We split the book into small chunks (pages/paragraphs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# --- 3. CREATE THE BRAIN (The "Memory") ---
print("--- 2. INDEXING DATA (Creating Vector Database) ---")
# This converts text -> numbers (Vectors) so the AI can search it.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# --- 4. CREATE THE TOOL ---
@tool
def search_pdf(query: str) -> str:
    """Useful for finding information inside the Neural Network PDF document."""
    # The agent uses this function to "read" the book
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

# --- 5. INITIALIZE THE AGENT ---
llm = ChatGroq(model="llama-3.1-8b-instant")
tools = [search_pdf]
agent = create_react_agent(llm, tools)

# --- 6. THE TEST ---
print("--- 3. AGENT THINKING ---")
# A question ONLY found in your PDF (Page 3/4 content)
query = "Explain how neural networks learn using the example of y = x^2 + x from the document."

response = agent.invoke({"messages": [("user", query)]})

print("\n✅ FINAL ANSWER:")
print(response['messages'][-1].content)