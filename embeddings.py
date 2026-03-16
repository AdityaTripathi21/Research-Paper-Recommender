import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests

CLEANED_PATH = "data/cleaned.jsonl"   # Your cleaned dataset
EMB_PATH = "data/embeddings.npy"      # Where we’ll save embeddings
META_PATH = "data/metadata.npy"       # save meta info

model = SentenceTransformer("all-MiniLM-L6-v2")     # load pretrained model from hugging face

# function returns array of abstracts and metadata of size n
def load_subset(n=10000):
    abstracts = []
    metas = []
    with open(CLEANED_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            paper = json.loads(line)
            abstracts.append(paper["abstract"])
            metas.append({
                "id": paper["id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "year": paper["year"],
            })
        return abstracts, metas
    
# embedding function    
def embed(n=10000, batch_size=256):
    abstracts, metas = load_subset(n)
    all_embeddings = []
    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i:i+batch_size]
        embs = model.encode(batch, show_progress_bar=True)  # tokenize text and return dense vector
        all_embeddings.append(embs)
    all_embeddings = np.vstack(all_embeddings)  # stack all embeddings (batch_size, d) into one matrix so it becomes (n, d)
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True) # normalize for cosine similarity, magnitude doesn't matter, only angle
    
    np.save(EMB_PATH, all_embeddings)   # embedding dimension is (n,d), where d is size of embedding vector
    np.save(META_PATH, metas)
    print(f"Saved embeddings to {EMB_PATH}")
    print(f"Saved metadata to {META_PATH}")

# build faiss index    
def faiss_index():
    embeddings = np.load(EMB_PATH).astype("float32")    # convert to float32 for FAISS
    dimension = embeddings.shape[1] # get d - 384
        
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine if normalized), no clusters created, just faster than numpy
    embeddings = np.ascontiguousarray(embeddings)   # embeddings stored contiguously in memory for FAISS
    index.add(embeddings) # type: ignore
    faiss.write_index(index, "data/faiss.index")    # save index to disk

# search embedding space and return top k results with metadata
def search(query, k=5):
    index = faiss.read_index("data/faiss.index")   # load index
    
    metas = np.load(META_PATH, allow_pickle=True)   # load metadata
    
    q_emb = model.encode([query])  # embed query
    
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)        # normalize query
    q_emb = q_emb.astype("float32")
    
    # D returns similarity scores and I returns indices of nearest neighbors
    D, I = index.search(q_emb, k)
    
    results = []
    for i in I[0]:      # I[0] because I returns a 2d array for multiple queries, however, we only have one query
        results.append(metas[i])
        
    return results

# context for LLM
def build_context(results):
    context = ""
    
    for i, paper in enumerate(results, 1):
        context += f"""
Paper {i}
Title: {paper['title']}
Year: {paper['year']}
Abstract: {paper['abstract']}
"""
    return context

# rag 
def rag_answer(query):
    results = search(query)
    context = build_context(results)
    
    
    prompt = f"""
You are a research assistant.

Use the following research papers to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


    

if __name__ == "__main__":
    # embed(n=10000)
    # faiss_index()
    query = input("Ask a research question: ")
    answer = rag_answer(query)
    print(answer)
