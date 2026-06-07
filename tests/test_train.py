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
