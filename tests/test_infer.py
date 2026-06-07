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
