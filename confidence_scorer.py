import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class ConfidenceScorer:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def score_tokens(self, question, context):
        # Combine question and context into a single input string
        combined = f"question: {question} context: {context}"

        # Tokenize input
        inputs = self.tokenizer(combined, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get token-level predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Softmax to get probabilities
        logits = outputs.logits  # shape: (1, seq_len, num_classes)
        probs = torch.softmax(logits, dim=-1)

        # We'll assume class 1 means "important", and use its probability as confidence
        importance_scores = probs[0, :, 1]  # (seq_len,)

        # Convert token IDs back to readable tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Combine tokens and scores
        token_score_pairs = list(zip(tokens, importance_scores.cpu().tolist()))

        return token_score_pairs
