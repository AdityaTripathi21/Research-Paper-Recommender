import json
import numpy as np
from sentence_transformers import SentenceTransformer

CLEANED_PATH = "data/cleaned.jsonl"   # Your cleaned dataset
EMB_PATH = "data/embeddings.npy"      # Where we’ll save embeddings
META_PATH = "data/metadata.npy"       # Optional: save meta info

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
                "year": paper["year"]
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
    
    np.save(EMB_PATH, all_embeddings)   # embedding dimension is (n,d), where d is size of embedding vector
    np.save(META_PATH, metas)
    print(f"Saved embeddings to {EMB_PATH}")
    print(f"Saved metadata to {META_PATH}")
    

if __name__ == "__main__":
    embed(n=10000)
