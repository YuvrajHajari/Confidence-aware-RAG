def group_tokens_with_scores(tokens, scores):
    grouped = []
    current_token = ""
    current_scores = []

    for token, score in zip(tokens, scores):
        if token.startswith("##"):
            current_token += token[2:]
            current_scores.append(score)
        else:
            if current_token:
                avg_score = sum(current_scores) / len(current_scores)
                grouped.append((current_token, avg_score))
            current_token = token
            current_scores = [score]

    # Append the last token
    if current_token:
        avg_score = sum(current_scores) / len(current_scores)
        grouped.append((current_token, avg_score))

    return grouped


from confidence_scorer import ConfidenceScorer

scorer = ConfidenceScorer()
question = "What planet is known as the Red Planet?"
context = "The solar system includes planets like Earth, Mars, Jupiter, and Saturn. Mars is often called the Red Planet due to its reddish appearance caused by iron oxide on its surface."

token_scores = scorer.score_tokens(question, context)

# Split tokens and scores
tokens = [token for token, _ in token_scores]
scores = [score for _, score in token_scores]

# Group them neatly
grouped = group_tokens_with_scores(tokens, scores)

# Print nicely
for token, score in grouped:
    print(f"{token:<15} -> {score:.3f}")

