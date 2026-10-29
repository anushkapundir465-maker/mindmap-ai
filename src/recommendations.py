from __future__ import annotations


TOPIC_RECOMMENDATIONS: dict[str, list[str]] = {
    "Natural Language Processing": [
        "Review how text is cleaned and converted into features before a model uses it.",
        "Compare sentiment analysis, translation, and text classification as NLP tasks.",
    ],
    "Tokenization": [
        "Practice splitting sentences into word and subword tokens.",
        "Review how token boundaries, punctuation, and vocabulary choices affect downstream models.",
    ],
    "TF-IDF": [
        "Work through term frequency and inverse document frequency on a tiny document collection.",
        "Compare a frequent word with a rare, informative word to see how the weight changes.",
    ],
    "Neural Networks": [
        "Trace one input through the input, hidden, and output layers.",
        "Review how weights and activation functions transform a prediction.",
    ],
    "Backpropagation": [
        "Follow the loss gradient backward through a small network using the chain rule.",
        "Review how gradients and a learning rate update model weights.",
    ],
    "LSTM": [
        "Draw how the cell state and input, forget, and output gates manage information.",
        "Review why gated memory can help preserve information across long sequences.",
    ],
    "GRU": [
        "Compare the reset and update gates and trace how they form a candidate hidden state.",
        "Contrast the GRU's simpler gate structure with an LSTM's cell state.",
    ],
}


def recommendations_for(topic: str, missing: list[str] | None = None) -> list[str]:
    advice = TOPIC_RECOMMENDATIONS.get(
        topic,
        ["Review the core terms, then explain the idea in your own words."],
    )
    if missing:
        return [f"Revisit **{concept}**: {advice[index % len(advice)]}" for index, concept in enumerate(missing)]
    return advice
