import pytest
import torch
from PIL import Image

from fast_vton.config import Config
from fast_vton.pipeline import FastVTONPipeline


@pytest.mark.slow
def test_inference_runs():
    if not torch.cuda.is_available():
        pytest.skip("cần GPU để chạy")

    predictor = FastVTONPipeline(Config())
    person = Image.new("RGB", (384, 512), (255, 255, 255))
    garment = Image.new("RGB", (224, 224), (200, 0, 0))
    agnostic = predictor.build_agnostic(person)
    result = predictor.try_on(person, agnostic, garment)
    assert result.shape == (1, 3, 512, 384)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
