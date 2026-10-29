from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text


TOPIC_CONCEPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "Natural Language Processing": {
        "language": ("language", "text", "human"),
        "preprocessing": ("preprocessing", "cleaning", "normalize", "normalization"),
        "applications": ("meaning", "sentiment", "translation", "classification"),
        "machine learning": ("machine", "learning", "model", "algorithm"),
    },
    "Tokenization": {
        "tokens": ("token", "tokens", "word", "words", "subword"),
        "boundaries": ("boundary", "boundaries", "split", "segment"),
        "punctuation": ("punctuation", "period", "comma"),
        "vocabulary": ("vocabulary", "vocab", "dictionary"),
    },
    "TF-IDF": {
        "term frequency": ("term", "frequency", "document"),
        "inverse document frequency": ("inverse", "rare", "corpus", "documents"),
        "weighting": ("weight", "weighting", "score"),
        "importance": ("important", "importance", "common", "distinctive"),
    },
    "Neural Networks": {
        "layers": ("layer", "layers", "hidden", "input", "output"),
        "weights": ("weight", "weights", "parameter", "parameters"),
        "activation": ("activation", "relu", "sigmoid", "nonlinear"),
        "prediction": ("prediction", "predict", "output", "learn"),
    },
    "Backpropagation": {
        "loss": ("loss", "error", "objective"),
        "gradient": ("gradient", "derivative", "partial"),
        "chain rule": ("chain", "rule", "layers"),
        "weight updates": ("update", "updates", "weights", "learning"),
    },
    "LSTM": {
        "memory cell": ("cell", "memory", "state"),
        "gates": ("gate", "gates", "input", "forget", "output"),
        "long dependencies": ("long", "dependency", "dependencies", "sequence"),
        "gradient flow": ("gradient", "vanishing", "flow"),
    },
    "GRU": {
        "gates": ("gate", "gates", "update", "reset"),
        "hidden state": ("hidden", "state", "candidate"),
        "sequence": ("sequence", "time", "temporal"),
        "efficiency": ("parameter", "parameters", "simpler", "efficient"),
    },
}


def keyword_coverage(reference: str, answer: str) -> float:
    """Return the fraction of distinct reference content words present in the answer."""
    reference_tokens = set(preprocess_text(reference))
    answer_tokens = set(preprocess_text(answer))
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & answer_tokens) / len(reference_tokens)


def tfidf_similarity(reference: str, answer: str) -> float:
    """Return cosine similarity of the two texts using a TF-IDF word/phrase baseline."""
    if not reference.strip() or not answer.strip():
        return 0.0
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
    )
    documents = [reference, answer]
    analyzer = vectorizer.build_analyzer()
    if not any(analyzer(document) for document in documents):
        return 0.0
    matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0, 0]
    return float(np.clip(similarity, 0.0, 1.0))


def missing_concepts(reference: str, answer: str, topic: str) -> list[str]:
    """Flag mapped concepts whose key terms occur in the reference but not the answer."""
    concepts = TOPIC_CONCEPTS.get(topic, {})
    reference_tokens = set(preprocess_text(reference, remove_stop_words=False))
    answer_tokens = set(preprocess_text(answer, remove_stop_words=False))
    missing: list[str] = []
    for concept, terms in concepts.items():
        mentioned_in_reference = reference_tokens.intersection(terms)
        mentioned_in_answer = answer_tokens.intersection(terms)
        if mentioned_in_reference and not mentioned_in_answer:
            missing.append(concept)
    return missing


def analyze_answer(reference: str, answer: str, topic: str) -> dict[str, object]:
    if not reference.strip() or not answer.strip():
        raise ValueError("Both the reference and student answer must contain text.")
    return {
        "keyword_coverage": keyword_coverage(reference, answer),
        "tfidf_similarity": tfidf_similarity(reference, answer),
        "missing_concepts": missing_concepts(reference, answer, topic),
    }


def load_semantic_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load the optional pretrained sentence model; return its failure explicitly."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, "Install sentence-transformers to enable pretrained semantic similarity."
    try:
        return SentenceTransformer(model_name), ""
    except Exception as error:
        return None, f"Could not download or load the pretrained model: {error}"
