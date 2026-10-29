# MindMap AI — Intelligent Learning Gap Detector

MindMap AI is a local Streamlit learning dashboard for comparing student explanations with reference material, practicing heuristic multiple-choice quizzes, and tracking quiz progress. It combines keyword coverage, a scikit-learn TF-IDF baseline, optional sentence embeddings, and a small PyTorch text classifier trained from the included labeled CSV.

## Requirements

- Python 3.11 or newer, below 3.15
- Windows PowerShell
- Internet access for installing packages and (optionally) downloading the sentence-transformer model

The current workspace has Python 3.14. The project requirements use current package ranges for Python 3.14 wheels (including PyTorch 2.9+ and scikit-learn 1.7+); dependency installation should be verified in the active environment before training. If a package does not publish a wheel for your Python installation, use Python 3.12 and recreate the virtual environment with that interpreter.

## Set up on Windows PowerShell

Run these commands from the project directory:

```powershell
py -0p
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for this terminal, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again. To use Python 3.12 instead, replace `py -3.14` with `py -3.12`.

## Train the PyTorch model

```powershell
.\.venv\Scripts\Activate.ps1
python train_model.py
```

The training script reads `data/sample_dataset.csv`, uses a stratified held-out split, trains a feed-forward PyTorch classifier, prints accuracy, macro precision, macro recall, macro F1, and a confusion matrix, then saves `models/mindmap_text_classifier.pt`. To select a different epoch count or paths:

```powershell
python train_model.py --epochs 100 --data data/sample_dataset.csv --output models/mindmap_text_classifier.pt
```

The labeled examples are small, hand-written teaching examples, not a validated assessment corpus. Held-out metrics are computed when you train; no accuracy is assumed or promised in advance. The model's two classes (`understanding` and `misconception`) and its confidence are educational demonstrations, not mastery scores.

On the first local run with the included dataset and fixed seed, the 13-example held-out test produced accuracy **0.231**, macro precision **0.225**, macro recall **0.238**, and macro F1 **0.226**; the confusion matrix (rows=true, columns=predicted; labels `misconception`, `understanding`) was `[[2, 4], [6, 1]]`. This weak result is reported rather than hidden: the tiny sample is not sufficient for dependable classification. Re-run training to see metrics for your installed versions and dataset.

## Launch the app

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local URL printed by Streamlit. Demo mode is selected initially and contains example dashboard activity and prefilled text. Switch to Personal mode to save quiz history locally in `data/progress.json`.

## Test

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app.py train_model.py src tests
```

## Features and how to try them

- **Dashboard:** review sample or personal quiz totals, averages, topic scores, and progress charts.
- **Answer analysis:** compare reference and student text using keyword coverage and TF-IDF; inspect topic-specific concept flags and review suggestions. Semantic similarity activates after its pretrained model downloads.
- **Practice quiz:** paste notes or upload a text-based PDF, generate experimental fill-in-the-blank choices, and submit answers. OCR is not included for scanned PDFs.
- **Progress & reports:** view local activity and download it as CSV; the answer-analysis page also exports its comparison and suggestions as a separate CSV. Demo activity is read-only.
- **PyTorch classifier:** train as above, then analyze an answer to see the classifier's predicted label and confidence.
- **Offline behavior:** core analysis, quiz generation, local history, and a trained classifier work locally. The sentence-transformer model needs an initial internet download; if unavailable, the app reports the issue and retains TF-IDF. Package installation also requires network access unless wheels are already cached.

## Project layout

```text
app.py
train_model.py
requirements.txt
README.md
.streamlit/config.toml
src/
  analysis.py
  demo.py
  model.py
  preprocessing.py
  quiz.py
  recommendations.py
  storage.py
data/
  sample_dataset.csv
  progress.json                 # created for personal activity
models/
  mindmap_text_classifier.pt    # created after successful training
tests/
  test_core.py
```
