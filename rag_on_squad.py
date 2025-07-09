from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Falcon model + tokenizer
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load SQuAD
dataset = load_dataset("squad", split="validation[:3]")  # small sample

# Loop through examples
for example in dataset:
    question = example["question"]
    context = example["context"]

    # Prepare input prompt
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    # Tokenize & move to model device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    # Decode and print result
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n=== QA Pair ===")
    print(f"Q: {question}")
    print(f"A: {answer[len(prompt):].strip()}")
