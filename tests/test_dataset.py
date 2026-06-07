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
