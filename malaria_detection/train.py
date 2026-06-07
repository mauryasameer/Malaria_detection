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
