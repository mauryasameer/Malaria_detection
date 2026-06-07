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
