# confidence_rag.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load the generator model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Device set to use {device}")

# Main function to generate answer from question + context
def generate_answer(question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    output = generator(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']
    return output
