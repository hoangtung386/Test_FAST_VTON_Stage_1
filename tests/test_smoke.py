import pytest

from fast_vton_test import FastVTONInference
from fast_vton_test.inference import DEFAULT_BUNDLE


def test_package_imports():
    assert FastVTONInference is not None
    assert isinstance(DEFAULT_BUNDLE, str)


@pytest.mark.slow
def test_inference_runs():
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        pytest.skip("cần GPU để chạy")
    predictor = FastVTONInference(DEFAULT_BUNDLE, device="cuda")
    person = Image.new("RGB", (384, 512), (255, 255, 255))
    garment = Image.new("RGB", (224, 224), (200, 0, 0))
    agnostic = predictor.build_agnostic(person)
    result = predictor.try_on(person, agnostic, garment)
    assert result.shape == (1, 3, 512, 384)
