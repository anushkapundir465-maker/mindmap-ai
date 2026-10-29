from __future__ import annotations

import re
import string

from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TOKENIZER = TreebankWordTokenizer()
STOP_WORDS = set(ENGLISH_STOP_WORDS)
PUNCTUATION_TABLE = str.maketrans({character: " " for character in string.punctuation})


def preprocess_text(text: str, remove_stop_words: bool = True) -> list[str]:
    """Lowercase and tokenize text, optionally removing common English stop words."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    normalized = text.lower().translate(PUNCTUATION_TABLE)
    tokens = [token for token in TOKENIZER.tokenize(normalized) if re.search(r"[a-z0-9]", token)]
    if remove_stop_words:
        tokens = [token for token in tokens if token not in STOP_WORDS]
    return tokens
