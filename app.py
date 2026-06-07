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
