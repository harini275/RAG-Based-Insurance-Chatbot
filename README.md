# RAG-Based-Insurance-Chatbot

This is a Retrieval-Augmented Generation (RAG) based chatbot that scrapes data from [Tata AIG's Health Insurance page](https://www.tataaig.com/health-insurance), stores embeddings in a FAISS vector database, and uses the Gemma 2B LLM from Ollama to answer user queries.

---

## 💡 Features

- Web scraping using BeautifulSoup
- Sentence embedding using SentenceTransformers
- FAISS vector similarity search for context retrieval
- Local LLM (Gemma 2B) with Ollama for generating answers
- Flask web app interface

---

## 🚀 How It Works

1. Scrapes the website for health insurance content.
2. Converts the text into embeddings using `all-MiniLM-L6-v2`.
3. Stores and searches text via FAISS.
4. Sends user question and retrieved context to `gemma:2b` via Ollama API.
5. Displays generated answer in a simple web interface.

---

## 🔧 Requirements

- Python 3.9+
- Flask
- FAISS
- SentenceTransformers
- BeautifulSoup
- Ollama (running Gemma 2B model locally)

---

## 🛠 Installation

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama and load the Gemma model
ollama serve
ollama pull gemma:2b

# Run the app
python app.py
