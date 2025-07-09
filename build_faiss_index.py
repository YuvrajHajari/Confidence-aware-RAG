from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# 1. Load a subset of the dataset
dataset = load_dataset("trivia_qa", "rc", split="train[:2000]")

# 2. Prepare the corpus
corpus = []
doc_ids = []

print("📦 Collecting documents...")
for i, sample in enumerate(dataset):
    results = sample.get("search_results", {})
    titles = results.get("title", [])
    descriptions = results.get("description", [])

    # Make sure both lists are same length
    for j in range(min(len(titles), len(descriptions))):
        title = titles[j].strip()
        desc = descriptions[j].strip()

        if title and desc:
            text = f"{title}. {desc}"
            corpus.append(text)
            doc_ids.append((i, title))

print(f"✅ Collected {len(corpus)} documents.")

if not corpus:
    raise ValueError("No documents were collected. Please check the dataset structure.")

# 3. Encode documents
print("🔍 Encoding documents...")
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)

# 4. Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
faiss.write_index(index, "trivia_faiss.index")

# 5. Save the corpus and IDs
with open("trivia_corpus.json", "w", encoding="utf-8") as f:
    json.dump({"corpus": corpus, "doc_ids": doc_ids}, f)

print("✅ FAISS index and document corpus saved.")
