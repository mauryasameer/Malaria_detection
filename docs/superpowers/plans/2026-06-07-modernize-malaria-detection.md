# Modernize Malaria Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old single-notebook `Malaria_detection` repo with a clean PyTorch + Gradio project (training script, inference module, Colab notebook, HF Spaces demo, modernized README with hero banner) suitable for linking from LinkedIn.

**Architecture:** A small `malaria_detection` Python package (`dataset.py`, `model.py`, `train.py`, `infer.py`) backs both a CLI training/inference flow and a Gradio web demo (`app.py`). Training runs on Colab GPU against the public NIH Malaria Cell Images dataset; the resulting checkpoint is loaded by `infer.py`/`app.py` for prediction. Tests use synthetic in-memory image folders (no dataset download required for CI).

**Tech Stack:** Python 3.11, PyTorch + torchvision (ResNet-18 transfer learning), Gradio, Pillow, pytest

---

## Pre-work: repo init

The working directory `/Users/sameermaurya/Downloads/dev/Malaria_detection` already contains `docs/` (spec + this plan) and `assets/banner.png`. It is not yet a git repo.

- [ ] **Step 1: Initialize git repo and create feature branch**

```bash
cd /Users/sameermaurya/Downloads/dev/Malaria_detection
git init
git checkout -b feature/modernize-project
```

- [ ] **Step 2: Add remote pointing at the existing GitHub repo**

```bash
git remote add origin https://github.com/mauryasameer/Malaria_detection.git
git fetch origin
```

Expected: fetch lists `master` (the old repo's default branch).

- [ ] **Step 3: Commit the spec/plan/banner already in place**

```bash
git add docs assets
git commit -m "chore: add design spec, plan, and hero banner"
```

---

## Task 1: Project scaffolding (requirements, license, gitignore)

**Files:**
- Create: `requirements.txt`
- Create: `LICENSE`
- Create: `.gitignore`
- Create: `malaria_detection/__init__.py`

- [ ] **Step 1: Write `requirements.txt`**

```
torch>=2.2
torchvision>=0.17
gradio>=4.0
pillow>=10.0
pytest>=8.0
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
outputs/
data/
*.egg-info/
.DS_Store
```

- [ ] **Step 3: Write `LICENSE`** (MIT, matches a typical personal portfolio repo)

```
MIT License

Copyright (c) 2026 Sameer Maurya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Create empty package marker**

```bash
mkdir -p malaria_detection tests
touch malaria_detection/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore LICENSE malaria_detection/__init__.py
git commit -m "chore: scaffold project (deps, license, gitignore, package)"
```

---

## Task 2: `model.py` — ResNet classifier builder

**Files:**
- Create: `malaria_detection/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch

from malaria_detection.model import build_model


def test_build_model_output_shape():
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    dummy_input = torch.randn(4, 3, 128, 128)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'malaria_detection.model'`

- [ ] **Step 3: Write the implementation**

```python
# malaria_detection/model.py
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add malaria_detection/model.py tests/test_model.py
git commit -m "feat: add ResNet-18 transfer-learning model builder"
```

---

## Task 3: Test fixtures for synthetic image folders

**Files:**
- Create: `tests/conftest.py`

This fixture builds a tiny on-disk `ImageFolder`-compatible directory of solid-color
PNGs so dataset/training tests don't need the real (multi-hundred-MB) dataset.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
# tests/conftest.py
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_fake_image_folder():
    def _make(root: Path, classes: list[str], images_per_class: int = 6) -> None:
        for class_name in classes:
            class_dir = root / class_name
            class_dir.mkdir(parents=True)
            for i in range(images_per_class):
                image = Image.new("RGB", (32, 32), color=(i * 10, i * 20, i * 30))
                image.save(class_dir / f"{class_name}_{i}.png")

    return _make
```

- [ ] **Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add synthetic image-folder fixture"
```

---

## Task 4: `dataset.py` — dataset loader and dataloaders

**Files:**
- Create: `malaria_detection/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
from malaria_detection.dataset import build_dataloaders, build_datasets


def test_build_datasets_splits_and_classes(tmp_path, make_fake_image_folder):
    make_fake_image_folder(tmp_path, ["parasitized", "uninfected"], images_per_class=10)

    train_dataset, val_dataset, classes = build_datasets(str(tmp_path), val_split=0.2, seed=42)

    assert classes == ["parasitized", "uninfected"]
    assert len(train_dataset) == 16
    assert len(val_dataset) == 4


def test_build_dataloaders_yield_correctly_shaped_batches(tmp_path, make_fake_image_folder):
    make_fake_image_folder(tmp_path, ["parasitized", "uninfected"], images_per_class=10)
    train_dataset, val_dataset, _ = build_datasets(str(tmp_path), val_split=0.2, seed=42)

    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, batch_size=4, num_workers=0)
    images, labels = next(iter(train_loader))

    assert images.shape[1:] == (3, 128, 128)
    assert labels.shape[0] == images.shape[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'malaria_detection.dataset'`

- [ ] **Step 3: Write the implementation**

```python
# malaria_detection/dataset.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGE_SIZE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_datasets(data_dir: str, val_split: float = 0.2, seed: int = 42):
    transform = build_transforms()
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset, full_dataset.classes


def build_dataloaders(train_dataset, val_dataset, batch_size: int = 32, num_workers: int = 2):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add malaria_detection/dataset.py tests/test_dataset.py
git commit -m "feat: add dataset loader and dataloader builders"
```

---

## Task 5: `infer.py` — checkpoint loading and single-image prediction

**Files:**
- Create: `malaria_detection/infer.py`
- Test: `tests/test_infer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infer.py
import torch
from PIL import Image

from malaria_detection.infer import predict
from malaria_detection.model import build_model


def test_predict_returns_known_label_and_valid_confidence():
    device = torch.device("cpu")
    model = build_model(num_classes=2, pretrained=False)
    model.to(device)
    model.eval()

    image = Image.new("RGB", (64, 64), color=(120, 40, 200))
    label, confidence = predict(model, image, device)

    assert label in ("parasitized", "uninfected")
    assert 0.0 <= confidence <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_infer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'malaria_detection.infer'`

- [ ] **Step 3: Write the implementation**

```python
# malaria_detection/infer.py
import argparse

import torch
from PIL import Image

from malaria_detection.dataset import build_transforms
from malaria_detection.model import build_model

CLASS_NAMES = ["parasitized", "uninfected"]


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model: torch.nn.Module, image: Image.Image, device: torch.device) -> tuple[str, float]:
    transform = build_transforms()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, predicted_index = torch.max(probabilities, dim=0)
    return CLASS_NAMES[predicted_index.item()], confidence.item()


def predict_from_path(checkpoint_path: str, image_path: str) -> tuple[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    image = Image.open(image_path)
    return predict(model, image, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a single cell image")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    parser.add_argument("image", help="Path to cell image")
    args = parser.parse_args()

    label, confidence = predict_from_path(args.checkpoint, args.image)
    print(f"{label} (confidence: {confidence:.2%})")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_infer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add malaria_detection/infer.py tests/test_infer.py
git commit -m "feat: add checkpoint loading and single-image prediction"
```

---

## Task 6: `train.py` — training loop and CLI

**Files:**
- Create: `malaria_detection/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
from malaria_detection.train import train_model


def test_train_model_writes_checkpoint_and_metrics(tmp_path, make_fake_image_folder):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    make_fake_image_folder(data_dir, ["parasitized", "uninfected"], images_per_class=10)

    metrics = train_model(str(data_dir), str(output_dir), epochs=1, batch_size=4)

    assert (output_dir / "model.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert metrics["classes"] == ["parasitized", "uninfected"]
    assert len(metrics["history"]) == 1
    assert "val_accuracy" in metrics["history"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'malaria_detection.train'`

- [ ] **Step 3: Write the implementation**

```python
# malaria_detection/train.py
import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim

from malaria_detection.dataset import build_dataloaders, build_datasets
from malaria_detection.model import build_model


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(
    data_dir: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = 42,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, classes = build_datasets(data_dir, seed=seed)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, batch_size=batch_size)

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "model.pt"
    metrics_path = output_path / "metrics.json"

    best_val_accuracy = 0.0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)

    metrics = {"classes": classes, "best_val_accuracy": best_val_accuracy, "history": history}
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the malaria cell classifier")
    parser.add_argument("--data-dir", required=True, help="Path to ImageFolder-structured dataset")
    parser.add_argument("--output-dir", default="outputs", help="Where to save checkpoint + metrics")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_model(args.data_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (takes a few seconds — trains 1 epoch on 16 tiny synthetic images)

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add malaria_detection/train.py tests/test_train.py
git commit -m "feat: add training loop, checkpointing, and metrics export"
```

---

## Task 7: Colab training notebook

**Files:**
- Create: `notebooks/train_colab.ipynb`

This notebook lets the user run real training on a Colab GPU against the NIH
Malaria Cell Images dataset, then download the resulting `model.pt` +
`metrics.json` to commit into the repo (per the spec's training approach).

- [ ] **Step 1: Write `notebooks/train_colab.ipynb`**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Malaria Detection — Colab Training\n",
    "\n",
    "Trains the ResNet-18 classifier from `malaria_detection/` on the NIH Malaria Cell Images dataset using a Colab GPU.\n",
    "\n",
    "**Steps:** clone repo → download dataset → install deps → train → download checkpoint."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!git clone https://github.com/mauryasameer/Malaria_detection.git\n",
    "%cd Malaria_detection\n",
    "!pip install -q -r requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download and extract the NIH Malaria Cell Images dataset (~340MB)\n",
    "!wget -q https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip -O cell_images.zip\n",
    "!unzip -q cell_images.zip -d data\n",
    "!ls data/cell_images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from malaria_detection.train import train_model\n",
    "\n",
    "metrics = train_model(\n",
    "    data_dir=\"data/cell_images\",\n",
    "    output_dir=\"outputs\",\n",
    "    epochs=10,\n",
    "    batch_size=64,\n",
    "    lr=1e-4,\n",
    ")\n",
    "metrics[\"best_val_accuracy\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download the trained checkpoint + metrics — commit these into the repo's outputs/ folder\n",
    "from google.colab import files\n",
    "\n",
    "files.download(\"outputs/model.pt\")\n",
    "files.download(\"outputs/metrics.json\")"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Commit**

```bash
mkdir -p notebooks
git add notebooks/train_colab.ipynb
git commit -m "docs: add Colab GPU training notebook"
```

---

## Task 8: Gradio demo app

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write `app.py`**

```python
# app.py
import os

import gradio as gr
import torch
from PIL import Image

from malaria_detection.infer import CLASS_NAMES, load_model, predict

CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT", "outputs/model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model(CHECKPOINT_PATH, DEVICE)


def classify(image: Image.Image) -> dict[str, float]:
    label, confidence = predict(MODEL, image, DEVICE)
    other_label = next(name for name in CLASS_NAMES if name != label)
    return {label: confidence, other_label: 1 - confidence}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Blood cell image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Malaria Detection — AI-Powered Cell Classification",
    description=(
        "Upload a microscope image of a single blood cell. "
        "The model classifies it as parasitized or uninfected."
    ),
)

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Smoke-test the app loads (requires a checkpoint at `outputs/model.pt` — skip if not yet trained)**

Run: `MODEL_CHECKPOINT=outputs/model.pt python app.py`
Expected: Gradio prints a local URL; if no checkpoint exists yet, this step is deferred until after Task 7's Colab run produces one — note that in the commit and move on.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Gradio demo app for live cell classification"
```

---

## Task 9: Modernized README

**Files:**
- Create: `README.md` (overwrites the old one)
- Modify/remove: retire `fastai.ipynb` and the old screenshot

- [ ] **Step 1: Move legacy notebook and screenshot out of the active tree**

```bash
mkdir -p legacy
git mv fastai.ipynb legacy/fastai.ipynb 2>/dev/null || true
git mv "Screenshot from 2019-03-18 16-10-08.png" legacy/screenshot-2019.png 2>/dev/null || true
```

(If `git mv` reports the files don't exist yet — they live in the old repo's
history, not this fresh local init. In that case skip this step; there is
nothing to move.)

- [ ] **Step 2: Write the new `README.md`**

```markdown
# Malaria Detection — AI-Powered Cell Classification

![Malaria Detection banner](assets/banner.png)

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-demo-orange)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Malaria kills hundreds of thousands of people every year, and diagnosis still
relies on a lab technician manually examining blood smears under a microscope.
This project fine-tunes a ResNet-18 image classifier to automatically tell
**parasitized** cells from **uninfected** ones — a step toward faster,
cheaper, more consistent malaria screening.

## Live demo

Try it on Hugging Face Spaces: **[malaria-detection demo](https://huggingface.co/spaces/mauryasameer/malaria-detection)**

Upload a microscope image of a single blood cell and get an instant
classification with a confidence score.

## How it works

```
NIH Malaria Cell Images dataset → ResNet-18 (transfer learning) → parasitized / uninfected
```

The model starts from ImageNet-pretrained weights and is fine-tuned on the
[NIH Malaria Cell Images dataset](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)
(~27,000 labeled cell images, two classes).

## Results

| Metric | Value |
|---|---|
| Validation accuracy | see `outputs/metrics.json` after training |
| Precision / Recall / F1 | see `outputs/metrics.json` after training |

Run the training notebook (below) to reproduce these numbers from scratch.

## Quickstart

```bash
git clone https://github.com/mauryasameer/Malaria_detection.git
cd Malaria_detection
pip install -r requirements.txt
```

**Run inference on a single image:**

```bash
python -m malaria_detection.infer outputs/model.pt path/to/cell.png
```

**Run the Gradio demo locally:**

```bash
python app.py
```

**Train from scratch:**

The full NIH dataset is too large to ship in the repo. Use the included Colab
notebook to train on a free GPU:

[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) — open in Colab,
run all cells, download the resulting `model.pt` + `metrics.json`.

Or train locally once you have the dataset extracted to `data/cell_images/`:

```bash
python -m malaria_detection.train --data-dir data/cell_images --output-dir outputs --epochs 10
```

## Project structure

```
malaria_detection/   # dataset, model, training, inference
notebooks/           # Colab GPU training notebook
app.py               # Gradio demo (deployed to HF Spaces)
tests/               # pytest suite (synthetic data, no dataset download needed)
```

## Tech stack

Python · PyTorch · torchvision · Gradio · pytest

## License

[MIT](LICENSE)
```

- [ ] **Step 3: Commit**

```bash
git add README.md legacy 2>/dev/null
git add README.md
git commit -m "docs: rewrite README with banner, live demo link, and quickstart"
```

---

## Task 10: Final verification and push

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite one more time**

Run: `pytest tests/ -v`
Expected: All tests PASS, zero failures

- [ ] **Step 2: Push the feature branch**

```bash
git push -u origin feature/modernize-project
```

- [ ] **Step 3: Open a PR against the existing repo's default branch**

```bash
gh pr create --title "Modernize project: PyTorch + Gradio rebuild with live demo" --body "$(cat <<'EOF'
## Summary
- Replace the 2020 FastAI notebook with a clean PyTorch + torchvision package (dataset/model/train/infer)
- Add a Gradio demo app deployable to Hugging Face Spaces
- Add a Colab GPU training notebook against the NIH Malaria Cell Images dataset
- Rewrite README with hero banner, live demo link, results, and quickstart
- Add pytest suite using synthetic image data (no dataset download required)

## Test plan
- [ ] `pytest tests/ -v` passes locally
- [ ] Train via `notebooks/train_colab.ipynb` on Colab, commit resulting `outputs/model.pt` + `outputs/metrics.json`
- [ ] Run `python app.py` locally with a real checkpoint and verify predictions
- [ ] Deploy `app.py` to a Hugging Face Space and update the README demo link
EOF
)"
```

- [ ] **Step 4: Note remaining manual follow-ups (not part of this PR's automated scope)**

After merge:
- Run the Colab notebook to produce a real `outputs/model.pt` + `outputs/metrics.json`, commit them
- Fill in the actual results table in `README.md` from `metrics.json`
- Create a Hugging Face Space, push `app.py` + `requirements.txt` + checkpoint there, update the demo link in `README.md`
- Replace placeholder result numbers and add a few real sample-prediction images to `assets/sample_images/`
