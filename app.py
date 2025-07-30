# app.py
from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

app = Flask(__name__)

# Load transformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Web scrape
scraped_texts = []
faiss_index = None
all_texts = []

def scrape_website(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 30]
    return paragraphs

def create_index(texts):
    embeddings = embedder.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, texts

def retrieve_context(query, index, texts, k=5):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return "\n".join([texts[i] for i in indices[0] if i < len(texts)])

import time

def query_llm(question, context, max_retries=3, retry_delay=2):
    prompt = f"""You are a helpful assistant. Use the CONTEXT to answer the QUESTION.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"""
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post("http://localhost:11434/api/generate", json={
                "model": "gemma:2b",
                "prompt": prompt,
                "stream": False
            }, timeout=15)
            res.raise_for_status()
            try:
                json_resp = res.json()
                return json_resp.get("response", "[No answer from model]")
            except Exception as e:
                print(f"❌ JSON decode error: {e}, response text: {res.text}")
                return "[No answer from model due to invalid response]"
        except requests.RequestException as e:
            print(f"❌ Error querying LLM API (attempt {attempt}): {e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return "[Error fetching response from model after retries]"

@app.route("/", methods=["GET", "POST"])
def home():
    answer = context = ""
    if request.method == "POST":
        user_question = request.form.get("question")
        context = retrieve_context(user_question, faiss_index, all_texts)
        answer = query_llm(user_question, context)
    return render_template("chat.html", answer=answer, context=context)

if __name__ == "__main__":
    url = "https://www.tataaig.com/health-insurance"
    scraped_texts = scrape_website(url)
    faiss_index, all_texts = create_index(scraped_texts)
    app.run(debug=True)
