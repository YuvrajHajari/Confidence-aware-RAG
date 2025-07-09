from datasets import load_dataset

dataset = load_dataset("trivia_qa", "rc", split="train[:1]")

print("Available keys:", dataset[0].keys())
print("\nSample content:\n")
for key in dataset[0]:
    print(f"{key} →", dataset[0][key])
