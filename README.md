ANU-RAG: Attentive Nugget-Driven Utilization for Retrieval-Augmented Generation 🚀
Official implementation of
“ANU-RAG: Confidence-Aware Token Selection for Enhanced RAG Efficiency and Accuracy”

📋 Overview
ANU-RAG is a novel framework that enhances Retrieval-Augmented Generation (RAG) by introducing confidence-based token pruning. It reduces the computational burden on the LLM while preserving critical context by:

🎯 Scoring individual tokens based on relevance to the user query

✂️ Pruning low-confidence tokens from retrieved passages

⚡ Significantly lowering the number of tokens sent to the LLM

📈 Maintaining (or improving) answer quality while reducing latency and token usage

💻 Installation
bash
Copy
Edit
# Clone the repository
git clone https://github.com/yourusername/ANU-RAG.git
cd ANU-RAG

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
🚀 Quickstart
python
Copy
Edit
# Run the confidence-aware RAG system
python run_confidence_rag.py

# For baseline (vanilla RAG without token pruning)
python vanilla_rag.py
You will be prompted to enter a question. The system will:

Retrieve top-k relevant documents using FAISS.

Score tokens using DistilBERT.

Prune low-confidence tokens and send only high-quality context to the LLM.

Generate a concise answer.

📚 Dataset Preparation
Dataset used: TriviaQA (RC subset)

To build the FAISS index:

bash
Copy
Edit
python build_faiss_index.py
This script loads 2000 training samples, extracts titles and descriptions, encodes them using SentenceTransformer, and saves a FAISS index and JSON corpus.

🔧 Implementation Details
Retriever: SentenceTransformer (MiniLM-L6-v2)

Confidence Scorer: DistilBERT (Token Classification)

LLM Generator: Falcon-7B-Instruct

Vector Store: FAISS (L2 IndexFlat)

Confidence Threshold: Adjustable (default = 0.7)

Query Flow:
Query ➝ FAISS ➝ Top-k docs ➝ Token Scoring ➝ Pruning ➝ LLM ➝ Answer

🧪 Results & Evaluation
Example Query:
❓ How does climate change affect polar bears?

Vanilla RAG:

Tokens Sent: ~300

LLM Gen Time: 2391.15 sec

Answer: “Climate change affects polar bears by…”

ANU-RAG:

Tokens Sent: ~100 (after pruning)

LLM Gen Time: 1874.24 sec

Answer: “Climate change can affect polar bears by…”

🟢 ANU-RAG delivers similar quality with ~22% faster inference.

⚙️ Key Features
✅ Pluggable confidence scorer module

🔄 Switch between ANU-RAG and Vanilla RAG easily

📊 Built-in timing for retrieval, scoring, generation

💬 Compatible with any HuggingFace-compatible LLM

📈 Performance Metrics
Metric	Vanilla RAG	ANU-RAG
Tokens Sent to LLM	High	Low
LLM Generation Time	Higher	Lower
Answer Quality	Comparable	Comparable or Better
Token Efficiency	❌	✅

📌 Limitations
The DistilBERT confidence scorer is not fine-tuned.

Not trained for multi-hop or multi-modal queries.

Currently supports only English.

📅 Future Work
Fine-tune confidence model on QA relevance datasets.

Add compression-aware retrievers (e.g., SALSA or EXIT).

Evaluate on other datasets like HotpotQA or NaturalQuestions.

🖼️ Appendix
Screenshots of dataset structure and FAISS index creation

Sample answers from both Vanilla and Confidence-Aware RAG

Code snippets for token scoring, pruning, and generation

📄 License
MIT License. See LICENSE file.

🙏 Acknowledgements
HuggingFace Datasets & Transformers

SentenceTransformers

Falcon LLM (tiiuae/falcon-7b-instruct)

FAISS by Facebook AI