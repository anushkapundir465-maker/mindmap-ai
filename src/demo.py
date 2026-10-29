from __future__ import annotations

from datetime import datetime, timezone


DEMO_ANALYSIS = {
    "reference": (
        "TF-IDF weights a term by its frequency in a document and its inverse document "
        "frequency across a corpus. Common words receive lower weights, while distinctive "
        "terms become more important for document representation. TF-IDF is a lexical "
        "weighting method, not a direct measure of semantic understanding."
    ),
    "answer": (
        "TF-IDF measures term frequency in a document. It uses inverse document frequency "
        "across a corpus so common words get lower weights and distinctive terms are important."
    ),
}


def demo_attempts() -> list[dict[str, object]]:
    topics = [
        ("Tokenization", 92),
        ("TF-IDF", 68),
        ("Neural Networks", 84),
        ("Backpropagation", 56),
        ("LSTM", 76),
        ("GRU", 88),
    ]
    now = datetime.now(timezone.utc)
    return [
        {
            "kind": "quiz",
            "topic": topic,
            "score": score,
            "total": 100,
            "correct": score,
            "source": "sample quiz",
            "timestamp": (now.replace(hour=9, minute=0, second=0) ).isoformat(timespec="seconds"),
        }
        for topic, score in topics
    ]
