from confidence_scorer import ConfidenceScorer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

# Load Falcon model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

# Initialize confidence scorer
scorer = ConfidenceScorer()

def filter_tokens(tokens_with_scores, threshold=0.7):
    """
    Filters tokens above the threshold and reconstructs context string.
    """
    filtered_tokens = [
        token for token, score in tokens_with_scores
        if score >= threshold and token not in ["[CLS]", "[SEP]"]
    ]
    return tokenizer.convert_tokens_to_string(filtered_tokens)

def filter_and_generate_answer(question, context, threshold=0.7):
    # Step 1: Get token confidence scores
    tokens_with_scores = scorer.score_tokens(question, context)

    # Step 2: Filter tokens based on score
    pruned_context = filter_tokens(tokens_with_scores, threshold)

    # Step 3: Prepare prompt
    prompt = f"Context: {pruned_context}\nQuestion: {question}\nAnswer:"

    # Step 4: Generate answer using Falcon
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
        generation_config=GenerationConfig(temperature=0.7)
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- Confidence RAG Output ---")
    print(f"Pruned Context: {pruned_context}")
    print(f"Answer: {answer.split('Answer:')[-1].strip()}\n")
