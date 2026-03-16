# 📄 Research Recommender

A local RAG (Retrieval-Augmented Generation) pipeline that semantically searches 10,000 arXiv paper abstracts and uses a local LLM to answer research questions — no API keys required.

---

## How It Works

1. **Embed** — paper abstracts are encoded into 384-dimensional vectors using `all-MiniLM-L6-v2`
2. **Index** — vectors are stored in a FAISS index for fast similarity search
3. **Search** — a user query is embedded and matched against the index to find the top-k most relevant papers
4. **Generate** — retrieved papers are passed as context to Mistral 7B (via Ollama) to synthesize an answer

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding Model | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector Index | FAISS (`IndexFlatIP`) |
| LLM | Mistral 7B via Ollama |
| Frontend | Streamlit |
| Dataset | arXiv (cleaned `.jsonl`) |
| Language | Python 3.13 |

---

## Project Structure

```
research-recommender/
├── app.py                  # Streamlit frontend
├── embeddings.py           # Embedding, search, and RAG logic
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
├── .gitignore
└── data/
    ├── cleaned.jsonl       # Preprocessed arXiv abstracts
    ├── embeddings.npy      # Saved embedding matrix (n, 384)
    ├── metadata.npy        # Saved paper metadata
    └── faiss.index         # FAISS vector index
```

---

## Setup

### Prerequisites
- Python 3.13
- [Ollama](https://ollama.com) installed and running

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/research-recommender.git
cd research-recommender
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull Mistral (one time only)
```bash
ollama pull mistral
```

### 5. Build the index (one time only)
In `embeddings.py`, uncomment the following lines in `__main__` and run:
```python
embed(n=10000)
faiss_index()
```
```bash
python embeddings.py
```
Then re-comment those lines.

---

## Running the App

Make sure Ollama is running (open the desktop app or run `ollama serve`), then:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Re-indexing

Re-run `embed()` and `faiss_index()` if you:
- Change or update the dataset
- Add new fields to the metadata
- Switch embedding models

---

## Future Improvements

- [ ] Similarity scores displayed alongside results
- [ ] Year range filter in sidebar
- [ ] Scale to 50k–100k papers
- [ ] Cross-encoder reranking
- [ ] Hybrid BM25 + semantic search
- [ ] Conversation memory for follow-up questions
