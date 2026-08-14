
from fast_vton import FastVTONPipeline
from fast_vton.config import Config, config_from_bundle
from fast_vton.postprocessing import tensor_to_pil
from fast_vton.preprocessing import AgnosticBuilder, image_to_tensor


def test_package_imports():
    assert FastVTONPipeline is not None


def test_config_defaults():
    config = Config()
    assert config.bundle_name == "Fast_VTON_full.pt"
    assert config.bundle_path == config.model_dir / config.bundle_name
    assert config.seg_clothing_ids == (4, 5, 6, 7)


def test_config_from_bundle_splits_path(tmp_path):
    bundle = tmp_path / "models" / "foo.pt"
    config = config_from_bundle(bundle)
    assert config.model_dir == tmp_path / "models"
    assert config.bundle_name == "foo.pt"


def test_image_to_tensor_range_and_shape():
    from PIL import Image

    image = Image.new("RGB", (384, 512), (255, 128, 0))
    tensor = image_to_tensor(image, width=384, height=512, device="cpu")
    assert tensor.shape == (1, 3, 512, 384)
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_tensor_to_pil_roundtrip():
    import torch

    tensor = torch.full((1, 3, 8, 8), 0.5)
    image = tensor_to_pil(tensor)
    assert image.size == (8, 8)
    assert image.mode == "RGB"


def test_agnostic_builder_clears():
    builder = AgnosticBuilder("mattmdjaga/segformer_b2_clothes", (4, 5, 6, 7), "cpu")
    assert builder._model is None
    builder.clear()  # must be safe even before loading
    assert builder._model is None
