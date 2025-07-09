# query_faiss.py

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the saved FAISS index
index = faiss.read_index("trivia_faiss.index")

# Load the original corpus (for retrieving the actual text)
with open("trivia_corpus.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    corpus = data["corpus"]
    doc_ids = data["doc_ids"]

# Load the same sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Take a user question
question = input("Ask a question: ")

# Encode the question
question_embedding = model.encode([question], convert_to_numpy=True)

# Search in the FAISS index
k = 5  # how many top results to retrieve
D, I = index.search(question_embedding, k)

# Show top-k results
print("\n🔍 Top-k Relevant Contexts:")
for rank, idx in enumerate(I[0]):
    print(f"\n#{rank+1}: {corpus[idx]}")
