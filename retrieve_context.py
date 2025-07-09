from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load FAISS index and id mapping
index = faiss.read_index("faiss_index/index.faiss")
with open("faiss_index/id_map.pkl", "rb") as f:
    id_map = pickle.load(f)

# Load dataset with documents
dataset = load_from_disk("faiss_index/docs_dataset")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Input question
question = input("\n❓ Enter your question: ")
question_embedding = model.encode(question, convert_to_numpy=True)
question_embedding = np.expand_dims(question_embedding, axis=0)

# Retrieve top-k documents
k = 5
D, I = index.search(question_embedding, k)

print(f"\n🔍 Top {k} Retrieved Contexts:")
for i, idx in enumerate(I[0]):
    doc = dataset[int(id_map[idx])]
    print(f"\n--- Result #{i+1} ---\n{doc['text']}")
