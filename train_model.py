from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from src.model import create_network, vectorize
from src.preprocessing import preprocess_text


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample_dataset.csv"
DEFAULT_MODEL_PATH = ROOT / "models" / "mindmap_text_classifier.pt"
LABELS = ["misconception", "understanding"]


def train(dataset_path: Path = DATA_PATH, output_path: Path = DEFAULT_MODEL_PATH, epochs: int = 100) -> dict[str, object]:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is not installed for this Python. Install the project requirements "
            "in the active virtual environment before training."
        ) from error

    if epochs < 1:
        raise ValueError("Epoch count must be at least 1.")
    data = pd.read_csv(dataset_path)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip()
    data = data[(data["text"] != "") & data["label"].isin(LABELS)]
    if data["label"].nunique() != 2 or data["label"].value_counts().min() < 4:
        raise ValueError("Provide at least four labeled examples for each class.")

    random.seed(42)
    torch.manual_seed(42)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"].tolist(),
        data["label"].tolist(),
        test_size=0.25,
        random_state=42,
        stratify=data["label"].tolist(),
    )
    vocabulary = {
        token: index
        for index, token in enumerate(
            sorted({token for text in train_texts for token in preprocess_text(text)})
        )
    }
    if not vocabulary:
        raise ValueError("Training text contains no usable vocabulary.")
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    x_train = torch.stack([vectorize(text, vocabulary) for text in train_texts])
    y_train = torch.tensor([label_to_index[label] for label in train_labels], dtype=torch.long)
    x_test = torch.stack([vectorize(text, vocabulary) for text in test_texts])

    model = create_network(len(vocabulary))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(x_test).argmax(dim=1).tolist()
    y_true = [label_to_index[label] for label in test_labels]
    accuracy = accuracy_score(y_true, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1]).tolist()
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=LABELS,
        zero_division=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "labels": LABELS,
            "metrics": {
                "accuracy": float(accuracy),
                "macro_precision": float(precision),
                "macro_recall": float(recall),
                "macro_f1": float(f1),
                "test_examples": len(test_labels),
                "confusion_matrix": matrix,
            },
        },
        output_path,
    )
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": matrix,
        "classification_report": report,
        "test_examples": len(test_labels),
        "output_path": output_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the MindMap AI PyTorch text classifier."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Labeled CSV with text,label columns.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Path for the trained model.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100).")
    args = parser.parse_args()
    try:
        results = train(args.data, args.output, args.epochs)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"Test examples: {results['test_examples']}")
    print(f"Accuracy:       {results['accuracy']:.3f}")
    print(f"Macro precision: {results['macro_precision']:.3f}")
    print(f"Macro recall:   {results['macro_recall']:.3f}")
    print(f"Macro F1:       {results['macro_f1']:.3f}")
    print("Confusion matrix (rows=true, columns=predicted; labels=misconception, understanding):")
    print(results["confusion_matrix"])
    print(results["classification_report"])
    print(f"Saved model to: {results['output_path']}")
    print("Metrics are from this held-out split of a small, pedagogical sample dataset.")


if __name__ == "__main__":
    main()
