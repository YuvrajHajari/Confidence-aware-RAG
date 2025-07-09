# vanilla_rag.py
import time
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from confidence_rag import generate_answer  # Reuse same generator

# Load FAISS index and document corpus
print("🔁 Loading FAISS index and document corpus...")
index = faiss.read_index("trivia_faiss.index")
with open("trivia_corpus.json", "r", encoding="utf-8") as f:
    corpus_data = json.load(f)
corpus = corpus_data["corpus"]

# Load encoder model
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Prompt user
question = input("🔎 Ask a question: ").strip()
if not question:
    print("❌ No question entered. Exiting.")
    exit()

# Step 1: Retrieval
start_retrieval = time.time()
query_embedding = encoder.encode([question])
top_k = 5
_, indices = index.search(np.array(query_embedding), top_k)
retrieved_contexts = [corpus[i] for i in indices[0]]
retrieval_time = time.time() - start_retrieval

# Step 2: Directly concatenate retrieved docs (no confidence scoring)
context = " ".join(retrieved_contexts)

# Step 3: Generation
print("\n🤖 Generating Answer...")
start_gen = time.time()
answer = generate_answer(question, context)
gen_time = time.time() - start_gen

# Results
print("\n✅ Final Answer:")
print(answer)

print("\n📊 Summary:")
print(f"- Retrieval Time: {retrieval_time:.2f} sec")
print(f"- LLM Generation Time: {gen_time:.2f} sec")
print("- Confidence Scoring: ❌ Skipped (Vanilla RAG)")
