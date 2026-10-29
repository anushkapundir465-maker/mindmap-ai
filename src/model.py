from __future__ import annotations

from pathlib import Path
from typing import Any

from src.preprocessing import preprocess_text


def create_network(vocabulary_size: int):
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise ImportError("PyTorch is required to load or train the text classifier.") from error

    class TextClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(vocabulary_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, features):
            return self.layers(features)

    return TextClassifier()


def vectorize(text: str, vocabulary: dict[str, int]):
    import torch

    features = torch.zeros(len(vocabulary), dtype=torch.float32)
    for token in preprocess_text(text):
        token_index = vocabulary.get(token)
        if token_index is not None:
            features[token_index] += 1.0
    if features.sum() > 0:
        features /= features.sum()
    return features


def load_trained_model(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch is not installed. Install dependencies to use the classifier.") from error
    if not path.exists():
        raise FileNotFoundError(f"No trained model exists at {path}. Run `python train_model.py` first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    vocabulary = checkpoint.get("vocabulary")
    labels = checkpoint.get("labels")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(vocabulary, dict) or not isinstance(labels, list) or state_dict is None:
        raise ValueError("The saved model checkpoint is missing its vocabulary, labels, or weights.")
    network = create_network(len(vocabulary))
    network.load_state_dict(state_dict)
    network.eval()
    return {"network": network, "vocabulary": vocabulary, "labels": labels}


def predict_text(model: dict[str, Any], text: str) -> tuple[str, float]:
    import torch

    if not text.strip():
        raise ValueError("Enter text before requesting a model prediction.")
    features = vectorize(text, model["vocabulary"]).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model["network"](features), dim=1)[0]
    class_index = int(torch.argmax(probabilities).item())
    return str(model["labels"][class_index]), float(probabilities[class_index].item())
