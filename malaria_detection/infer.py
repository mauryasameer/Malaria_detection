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
