from datasets import load_dataset

# Load SQuAD v1.1 (you can change to "squad_v2" if needed)
dataset = load_dataset("squad")

# Print a few examples
for i in range(3):
    print(f"Question: {dataset['train'][i]['question']}")
    print(f"Context: {dataset['train'][i]['context']}")
    print(f"Answer: {dataset['train'][i]['answers']['text'][0]}")
    print()
