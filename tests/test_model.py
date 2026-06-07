import torch

from malaria_detection.model import build_model


def test_build_model_output_shape():
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    dummy_input = torch.randn(4, 3, 128, 128)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 2)
