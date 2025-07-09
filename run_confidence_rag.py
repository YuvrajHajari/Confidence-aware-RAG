# run_confidence_rag.py

import time
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from confidence_scorer import ConfidenceScorer
from confidence_rag import generate_answer

# Load FAISS index and corpus
print("🔁 Loading FAISS index and document corpus...")
index = faiss.read_index("trivia_faiss.index")
with open("trivia_corpus.json", "r", encoding="utf-8") as f:
    corpus_data = json.load(f)
corpus = corpus_data["corpus"]

# Load encoder model
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Confidence scorer
scorer = ConfidenceScorer()

# Prompt user
question = input("🔎 Ask a question: ").strip()
if not question:
    print("❌ No question entered. Exiting.")
    exit()

# Step 1: Retrieval with timer
print("\n📡 Retrieving relevant contexts...")
start = time.time()
query_embedding = encoder.encode([question])
top_k = 5
_, indices = index.search(np.array(query_embedding), top_k)
retrieved_contexts = [corpus[i] for i in indices[0]]
retrieval_time = time.time() - start

# Join retrieved contexts for scoring
full_context = " ".join(retrieved_contexts)

# Step 2: Confidence Scoring with timer
print("\n🔍 Running confidence scoring...")
start = time.time()
token_scores = scorer.score_tokens(question, full_context)
scoring_time = time.time() - start

# Prune low-confidence tokens
threshold = 0.5  # Original working threshold
pruned_tokens = [tok for tok, score in token_scores if score >= threshold]
pruned_context = "".join(pruned_tokens)

print("\n📘 Pruned Context:")
print(pruned_context)

# Step 3: Answer Generation with timer
print("\n🤖 Generating Answer...")
start = time.time()
answer = generate_answer(question, pruned_context)
generation_time = time.time() - start

print("\n✅ Final Answer:")
print(answer)

# Step 4: Report timings
print("\n📊 Summary:")
print(f"- Retrieval Time: {retrieval_time:.2f} sec")
print(f"- Confidence Scoring Time: {scoring_time:.2f} sec")
print(f"- LLM Generation Time: {generation_time:.2f} sec")


'''
import time
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from confidence_scorer import ConfidenceScorer
from confidence_rag import generate_answer


# Load FAISS index and corpus
print("🔁 Loading FAISS index and document corpus...")
index = faiss.read_index("trivia_faiss.index")
with open("trivia_corpus.json", "r", encoding="utf-8") as f:
    corpus_data = json.load(f)
corpus = corpus_data["corpus"]

# Load encoder model
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Confidence scorer
scorer = ConfidenceScorer()

# Prompt user
question = input("🔎 Ask a question: ").strip()
if not question:
    print("❌ No question entered. Exiting.")
    exit()

# Step 1: Retrieval with timer
print("\n📡 Retrieving relevant contexts...")
start = time.time()
query_embedding = encoder.encode([question])
top_k = 5
_, indices = index.search(np.array(query_embedding), top_k)
retrieved_contexts = [corpus[i] for i in indices[0]]
retrieval_time = time.time() - start

# Join retrieved contexts for scoring
full_context = " ".join(retrieved_contexts)

# Step 2: Confidence Scoring with timer
print("\n🔍 Running confidence scoring...")
start = time.time()
token_scores = scorer.score_tokens(question, full_context)
scoring_time = time.time() - start

# Prune low-confidence tokens
threshold = 0.7
pruned_tokens = [tok for tok, score in token_scores if score >= threshold]
pruned_context = "".join(pruned_tokens)


print("\n📘 Pruned Context:")
print(pruned_context)

# Step 3: Answer Generation with timer
print("\n🤖 Generating Answer...")
start = time.time()
answer = generate_answer(question, pruned_context)
generation_time = time.time() - start

print("\n✅ Final Answer:")
print(answer)

# Step 4: Report timings
print("\n📊 Summary:")
print(f"- Retrieval Time: {retrieval_time:.2f} sec")
print(f"- Confidence Scoring Time: {scoring_time:.2f} sec")
print(f"- LLM Generation Time: {generation_time:.2f} sec")
'''