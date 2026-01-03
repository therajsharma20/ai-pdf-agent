# 📄 AI PDF Agent (RAG Application)

### 🚀 **Live Demo:** [Click Here to Chat with your PDF](https://ai-pdf-agent.streamlit.app)

## 📌 Overview
This is a **Retrieval-Augmented Generation (RAG)** application built with Python. It allows users to upload a PDF document and ask questions about its content. The AI retrieves relevant context from the document to provide accurate, source-based answers, solving the common "hallucination" problem of LLMs.

## 🛠️ Tech Stack
* **LLM:** Llama 3 (via Groq API) - *High-speed inference*
* **Vector Database:** ChromaDB - *Semantic search & storage*
* **Orchestration:** LangChain - *RAG pipeline management*
* **Frontend:** Streamlit - *Interactive user interface*
* **Language:** Python 3.10+

## ✨ Features
* **Document Ingestion:** Uploads and splits PDF text into manageable chunks.
* **Vector Embeddings:** Converts text into vector representations for semantic search.
* **Contextual Q&A:** Retrieves the most relevant document sections to answer queries.
* **Secure:** "Bring Your Own Key" architecture ensures user privacy.

## 📦 How to Run Locally
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/therajsharma20/ai-pdf-agent.git](https://github.com/therajsharma20/ai-pdf-agent.git)
    cd ai-pdf-agent
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
